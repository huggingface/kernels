#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

void __KERNEL_NAME_NORMALIZED__(torch::Tensor &out, torch::Tensor const &input) {
  TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat ||
                  input.scalar_type() == torch::kHalf,
              "only float32 and float16 supported");
  TORCH_CHECK(input.sizes() == out.sizes(), "Tensors must have same shape");
  TORCH_CHECK(input.scalar_type() == out.scalar_type(), "Tensors must have same dtype");
  TORCH_CHECK(input.device() == out.device(), "Tensors must be on same device");

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    int numThreads = input.numel();

    NSError *error = nil;
    id<MTLLibrary> library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library: ",
                error.localizedDescription.UTF8String);

    std::string kernel_name = std::string("__KERNEL_NAME_NORMALIZED___forward_kernel_") +
        (input.scalar_type() == torch::kFloat ? "float" : "half");
    id<MTLFunction> func = [library newFunctionWithName:
        [NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(func, "Failed to create function: ", kernel_name.c_str());

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:getMTLBufferStorage(input)
                  offset:input.storage_offset() * input.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(out)
                  offset:out.storage_offset() * out.element_size()
                 atIndex:1];

      NSUInteger tgSize = MIN(pso.maxTotalThreadsPerThreadgroup, (NSUInteger)numThreads);
      [encoder dispatchThreads:MTLSizeMake(numThreads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
      [encoder endEncoding];
      torch::mps::commit();
    });
  }
}
