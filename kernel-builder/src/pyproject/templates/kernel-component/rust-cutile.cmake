if(GPU_LANG STREQUAL "CUDA")
    rust_kernel_component(RUST_KERNEL_LIBS RUST_KERNEL_TARGETS
        NAME {{ name }}
        MANIFEST_PATH "{{ manifest_path }}"
        CUDA_FRONTEND cutile
    )
endif()
