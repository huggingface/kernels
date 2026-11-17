#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// Include metal-cpp headers from system
#include <Metal/Metal.hpp>
#include <Foundation/NSSharedPtr.hpp>

#include <dispatch/dispatch.h>
#include <torch/mps.h>
#include <torch/torch.h>

// C interface from metallib_loader.mm
extern "C" void* loadEmbeddedMetalLibrary(void* device, const char** errorMsg);
extern "C" void* getMPSDevice();

namespace {

struct ReluEncodeArgs {
  MTL::CommandBuffer* commandBuffer;
  MTL::ComputePipelineState* pipelineState;
  MTL::Buffer* inputBuffer;
  NS::UInteger inputOffset;
  MTL::Buffer* outputBuffer;
  NS::UInteger outputOffset;
  NS::UInteger numel;
  bool encoded;
};

// Runs on PyTorch's MPS dispatch queue; reports failure through `encoded`.
void encodeRelu(void* ctx) {
  auto* args = static_cast<ReluEncodeArgs*>(ctx);

  MTL::ComputeCommandEncoder* encoder = args->commandBuffer->computeCommandEncoder();
  if (encoder == nullptr) {
    return;
  }

  encoder->setComputePipelineState(args->pipelineState);
  encoder->setBuffer(args->inputBuffer, args->inputOffset, 0);
  encoder->setBuffer(args->outputBuffer, args->outputOffset, 1);

  NS::UInteger threadGroupSize = args->pipelineState->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > args->numel) {
    threadGroupSize = args->numel;
  }

  const MTL::Size gridSize = MTL::Size::Make(args->numel, 1, 1);
  const MTL::Size threadsPerThreadgroup = MTL::Size::Make(threadGroupSize, 1, 1);
  encoder->dispatchThreads(gridSize, threadsPerThreadgroup);
  encoder->endEncoding();

  torch::mps::commit();
  args->encoded = true;
}

MTL::Buffer* getMTLBuffer(const torch::Tensor& tensor) {
  return reinterpret_cast<MTL::Buffer*>(const_cast<void*>(tensor.storage().data()));
}

NS::String* makeNSString(const std::string& value) {
  return NS::String::string(value.c_str(), NS::StringEncoding::UTF8StringEncoding);
}

MTL::Library* loadLibrary(MTL::Device* device) {
  const char* errorMsg = nullptr;
  void* library = loadEmbeddedMetalLibrary(reinterpret_cast<void*>(device), &errorMsg);

  TORCH_CHECK(library != nullptr, "Failed to create Metal library from embedded data: ",
              errorMsg ? errorMsg : "Unknown error");

  if (errorMsg) {
    free(const_cast<char*>(errorMsg));
  }

  return reinterpret_cast<MTL::Library*>(library);
}

} // namespace

void dispatchReluKernel(const torch::Tensor& input, torch::Tensor& output) {
  if (input.numel() == 0) {
    return;
  }

  // Use PyTorch's MPS device (a borrowed reference, not owned)
  MTL::Device* device = reinterpret_cast<MTL::Device*>(getMPSDevice());
  TORCH_CHECK(device != nullptr, "Failed to get MPS device");

  MTL::Library* libraryPtr = reinterpret_cast<MTL::Library*>(loadLibrary(device));
  NS::SharedPtr<MTL::Library> library = NS::TransferPtr(libraryPtr);

  const std::string kernelName =
      std::string("relu_forward_kernel_") + (input.scalar_type() == torch::kFloat ? "float" : "half");
  NS::SharedPtr<NS::String> kernelNameString = NS::TransferPtr(makeNSString(kernelName));

  NS::SharedPtr<MTL::Function> computeFunction =
      NS::TransferPtr(library->newFunction(kernelNameString.get()));
  TORCH_CHECK(computeFunction.get() != nullptr, "Failed to create Metal function for ", kernelName);

  NS::Error* pipelineError = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> pipelineState =
      NS::TransferPtr(device->newComputePipelineState(computeFunction.get(), &pipelineError));
  TORCH_CHECK(pipelineState.get() != nullptr,
              "Failed to create compute pipeline state: ",
              pipelineError ? pipelineError->localizedDescription()->utf8String() : "Unknown error");

  auto* inputBuffer = getMTLBuffer(input);
  auto* outputBuffer = getMTLBuffer(output);
  TORCH_CHECK(inputBuffer != nullptr, "Input buffer is null");
  TORCH_CHECK(outputBuffer != nullptr, "Output buffer is null");

  auto* commandBuffer = reinterpret_cast<MTL::CommandBuffer*>(torch::mps::get_command_buffer());
  TORCH_CHECK(commandBuffer != nullptr, "Failed to retrieve command buffer reference");

  auto serialQueue = static_cast<dispatch_queue_t>(torch::mps::get_dispatch_queue());

  ReluEncodeArgs args{
      commandBuffer,
      pipelineState.get(),
      inputBuffer,
      static_cast<NS::UInteger>(input.storage_offset() * input.element_size()),
      outputBuffer,
      static_cast<NS::UInteger>(output.storage_offset() * output.element_size()),
      static_cast<NS::UInteger>(input.numel()),
      false,
  };
  dispatch_sync_f(serialQueue, &args, encodeRelu);
  TORCH_CHECK(args.encoded, "Failed to create compute command encoder");
}

void relu(torch::Tensor& out, const torch::Tensor& input) {
  TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat || input.scalar_type() == torch::kHalf,
              "Unsupported data type: ", input.scalar_type());

  TORCH_CHECK(input.sizes() == out.sizes(),
              "Tensors must have the same shape. Got input shape: ",
              input.sizes(), " and output shape: ", out.sizes());

  TORCH_CHECK(input.scalar_type() == out.scalar_type(),
              "Tensors must have the same data type. Got input dtype: ",
              input.scalar_type(), " and output dtype: ", out.scalar_type());

  TORCH_CHECK(input.device() == out.device(),
              "Tensors must be on the same device. Got input device: ",
              input.device(), " and output device: ", out.device());

  dispatchReluKernel(input, out);
}
