#include <torch/extension.h>
#include "quat.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("q2RCUDA", &q2RCUDA);
  m.def("qMultCUDA", &qMultCUDA);
}
