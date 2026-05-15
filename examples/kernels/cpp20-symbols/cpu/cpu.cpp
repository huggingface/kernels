#include <array>
#include <charconv>
#include <stdexcept>

#include <torch/all.h>

// std::to_chars(char*, char*, double) is a floating-point overload that
// requires GLIBCXX_3.4.29, introduced in GCC 11. We use this to verify
// that manylinux_2_28 kernels build correctly: the Red Hat toolset
// statically links the newer libstdc++ symbols that exceed the system
// GLIBCXX_3.4.25 ceiling of AlmaLinux 8 / RHEL 8.
torch::Tensor float_to_chars(torch::Tensor const &input) {
    std::array<char, 32> buf;
    auto [ptr, ec] = std::to_chars(buf.begin(), buf.end(), input.item<double>());
    if (ec != std::errc{})
        throw std::runtime_error("to_chars failed");
    return input;
}
