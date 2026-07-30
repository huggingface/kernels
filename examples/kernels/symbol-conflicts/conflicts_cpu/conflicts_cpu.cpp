#include <cstddef>

#include <torch/all.h>


namespace test {

static int64_t global_counter = 0;

int64_t conflicts(torch::Tensor const &input) {
  return global_counter++;
}

int64_t conflicts_dynamic(torch::Tensor const &input) {
  static int64_t func_counter = 0;
  return func_counter++;
}

struct Conflicts {
  static int value() {
    static int64_t class_counter = 0;
    return class_counter++;
  }
};

static Conflicts conflicts_obj;

int64_t conflicts_class(torch::Tensor const &input) {
  return conflicts_obj.value();
}

}
