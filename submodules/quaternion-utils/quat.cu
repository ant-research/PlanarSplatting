#include "quat.h"
#include "quat_convert.h"
#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>


torch::Tensor q2RCUDA(const torch::Tensor& rotations)
{
  const int P = rotations.size(0);
  auto float_opts = rotations.options().dtype(torch::kFloat32);
  torch::Tensor transMat = torch::full({P, 3, 3}, 0.0, float_opts);
  
  QUAT_CONVERT::q2R(P, rotations.contiguous().data<float>(), transMat.contiguous().data<float>());


  return transMat;
}

torch::Tensor qMultCUDA(const torch::Tensor& rotations1, const torch::Tensor& rotations2)
{
  const int P = rotations1.size(0);
  auto float_opts = rotations1.options().dtype(torch::kFloat32);
  torch::Tensor rotations3 = torch::full({P, 4}, 0.0, float_opts);
  
  QUAT_CONVERT::qMult(P, rotations1.contiguous().data<float>(), rotations2.contiguous().data<float>(), rotations3.contiguous().data<float>());

  return rotations3;
}


