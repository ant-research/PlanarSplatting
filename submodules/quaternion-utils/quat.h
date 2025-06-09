#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
torch::Tensor q2RCUDA(const torch::Tensor& rotations);

torch::Tensor qMultCUDA(const torch::Tensor& rotations1, const torch::Tensor& rotations2);