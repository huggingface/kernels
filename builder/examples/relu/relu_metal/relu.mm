#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Include the auto-generated header with embedded metallib
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}


torch::Tensor &dispatchReluKernel(torch::Tensor const &input,
                                  torch::Tensor &output) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    int numThreads = input.numel();

    // Load the embedded Metal library from memory
    NSError *error = nil;
    id<MTLLibrary> customKernelLibrary = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(customKernelLibrary,
                "Failed to create Metal library from embedded data: ",
                error.localizedDescription.UTF8String);

    std::string kernel_name =
        std::string("relu_forward_kernel_") +
        (input.scalar_type() == torch::kFloat ? "float" : "half");
    id<MTLFunction> customReluFunction = [customKernelLibrary
        newFunctionWithName:[NSString
                                stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(customReluFunction,
                "Failed to create function state object for ",
                kernel_name.c_str());

    id<MTLComputePipelineState> reluPSO =
        [device newComputePipelineStateWithFunction:customReluFunction
                                              error:&error];
    TORCH_CHECK(reluPSO, error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> computeEncoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      [computeEncoder setComputePipelineState:reluPSO];
      [computeEncoder setBuffer:getMTLBufferStorage(input)
                         offset:input.storage_offset() * input.element_size()
                        atIndex:0];
      [computeEncoder setBuffer:getMTLBufferStorage(output)
                         offset:output.storage_offset() * output.element_size()
                        atIndex:1];

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

      NSUInteger threadGroupSize = reluPSO.maxTotalThreadsPerThreadgroup;
      if (threadGroupSize > numThreads) {
        threadGroupSize = numThreads;
      }
      MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

      [computeEncoder dispatchThreads:gridSize
                threadsPerThreadgroup:threadgroupSize];

      [computeEncoder endEncoding];

      torch::mps::commit();
    });
  }

  return output;
}

void relu(torch::Tensor &out, torch::Tensor const &input) {
  TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat ||
                  input.scalar_type() == torch::kHalf,
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
