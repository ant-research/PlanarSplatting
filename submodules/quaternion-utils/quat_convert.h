#ifndef CUDA_QUAT_CONVERT_H_INCLUDED
#define CUDA_QUAT_CONVERT_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace QUAT_CONVERT
{
	void q2R(int P, float* rotations, float* transMat);

	void qMult(int P, float* rotations1, float* rotations2, float* rotations3);
}


#endif
