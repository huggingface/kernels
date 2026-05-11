#include <array>
#include <charconv>
#include <stdexcept>

#include <torch/all.h>

// std::to_chars(char*, char*, double) is a floating-point overload that
// requires GLIBCXX_3.4.29, introduced in GCC 11. The manylinux_2_28 policy
// only allows up to GLIBCXX_3.4.25 (the GCC 8.5 system libstdc++ on
// AlmaLinux 8 / RHEL 8), so this symbol is prohibited and should be
// rejected by the ABI check.
torch::Tensor invalid_cpp_symbol(torch::Tensor const &input) {
    std::array<char, 32> buf;
    auto [ptr, ec] = std::to_chars(buf.begin(), buf.end(), input.item<double>());
    if (ec != std::errc{})
        throw std::runtime_error("to_chars failed");
    return input;
}
