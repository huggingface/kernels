import re

BUILD_VARIANT_REGEX = re.compile(r"^(torch\d+\d+|torch-(cpu|cuda|metal|rocm|xpu))")
