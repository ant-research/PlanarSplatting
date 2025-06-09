#include "stdio.h"
#include <iostream>
#include "quat_convert.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


__global__ void q2RCUDA(int P, const glm::vec4* rotations, float* transMats)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	
	// Compute transformation matrix
	glm::mat3 R = quat_to_rotmat(rotations[idx]);

	float3 *T_ptr = (float3*)transMats;
	T_ptr[idx * 3 + 0] = {R[0][0], R[0][1], R[0][2]};
	T_ptr[idx * 3 + 1] = {R[1][0], R[1][1], R[1][2]};
	T_ptr[idx * 3 + 2] = {R[2][0], R[2][1], R[2][2]}; 
}

__global__ void qMultCUDA(int P, const glm::vec4* rotations1, const glm::vec4* rotations2, glm::vec4* rotations3)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	
	// Compute transformation matrix
	glm::vec4 res = quat_mult_quat(rotations1[idx], rotations2[idx]);

	rotations3[idx].w = res.w;
    rotations3[idx].x = res.x;
    rotations3[idx].y = res.y;
    rotations3[idx].z = res.z;
}

void QUAT_CONVERT::q2R(int P, float* rotations, float* transMat){
    q2RCUDA<< <(P + 255) / 256, 256 >> > (P, (glm::vec4*)rotations,transMat);
}

void QUAT_CONVERT::qMult(int P, float* rotations1, float* rotations2, float* rotations3){
    qMultCUDA<< <(P + 255) / 256, 256 >> > (P, (glm::vec4*)rotations1, (glm::vec4*)rotations2, (glm::vec4*)rotations3);
}