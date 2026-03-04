#import <Metal/Metal.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

// C++ interface to load the embedded metallib without exposing ObjC types
extern "C" {
  void* loadEmbeddedMetalLibrary(void* device, const char** errorMsg) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    NSError* error = nil;

    id<MTLLibrary> library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(mtlDevice, &error);

    if (!library && errorMsg && error) {
      *errorMsg = strdup([error.localizedDescription UTF8String]);
    }

    // Manually retain since we're not using ARC
    // The caller will wrap in NS::TransferPtr which assumes ownership
    if (library) {
      [library retain];
    }
    return (__bridge void*)library;
  }

  // Get PyTorch's MPS device (returns id<MTLDevice> as void*)
  void* getMPSDevice() {
    return (__bridge void*)at::mps::MPSDevice::getInstance()->device();
  }

  // Get PyTorch's current MPS command queue (returns id<MTLCommandQueue> as void*)
  void* getMPSCommandQueue() {
    return (__bridge void*)at::mps::getCurrentMPSStream()->commandQueue();
  }

  // Get the MPS stream's command encoder (returns id<MTLComputeCommandEncoder> as void*).
  // Uses PyTorch's encoder lifecycle management (kernel coalescing).
  void* getMPSCommandEncoder() {
    return (__bridge void*)at::mps::getCurrentMPSStream()->commandEncoder();
  }

  // Commit the current command buffer and continue with a new one.
  void mpsSynchronize() {
    at::mps::getCurrentMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
  }

  // Dispatch a block on the MPS stream's serial queue.
  void mpsDispatchSync(void (*block)(void* ctx), void* ctx) {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
      block(ctx);
    });
  }
}
