#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// Include metal-cpp headers from system
#include <Metal/Metal.hpp>
#include <Foundation/NSSharedPtr.hpp>

#include <torch/torch.h>

// C interface from metallib_loader.mm
extern "C" void* loadEmbeddedMetalLibrary(void* device, const char** errorMsg);
extern "C" void* getMPSDevice();
extern "C" void* getMPSCommandQueue();

namespace {

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
  // Use PyTorch's MPS device and command queue (these are borrowed references, not owned)
  MTL::Device* device = reinterpret_cast<MTL::Device*>(getMPSDevice());
  TORCH_CHECK(device != nullptr, "Failed to get MPS device");

  MTL::CommandQueue* commandQueue = reinterpret_cast<MTL::CommandQueue*>(getMPSCommandQueue());
  TORCH_CHECK(commandQueue != nullptr, "Failed to get MPS command queue");

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

  // Don't use SharedPtr for command buffer/encoder - they're managed by PyTorch's command queue
  MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
  TORCH_CHECK(commandBuffer != nullptr, "Failed to create Metal command buffer");

  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  TORCH_CHECK(encoder != nullptr, "Failed to create compute command encoder");

  encoder->setComputePipelineState(pipelineState.get());

  auto* inputBuffer = getMTLBuffer(input);
  auto* outputBuffer = getMTLBuffer(output);
  TORCH_CHECK(inputBuffer != nullptr, "Input buffer is null");
  TORCH_CHECK(outputBuffer != nullptr, "Output buffer is null");

  encoder->setBuffer(inputBuffer, input.storage_offset() * input.element_size(), 0);
  encoder->setBuffer(outputBuffer, output.storage_offset() * output.element_size(), 1);

  const NS::UInteger totalThreads = input.numel();
  NS::UInteger threadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > totalThreads) {
    threadGroupSize = totalThreads;
  }

  const MTL::Size gridSize = MTL::Size::Make(totalThreads, 1, 1);
  const MTL::Size threadsPerThreadgroup = MTL::Size::Make(threadGroupSize, 1, 1);

  encoder->dispatchThreads(gridSize, threadsPerThreadgroup);
  encoder->endEncoding();

  commandBuffer->commit();
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
