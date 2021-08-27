#include "head.h"


void h2_mpo::init1(int a,int b)
{
	cudaMalloc((void**)&mpo1,sizeof(float)*a*b);
}
void h2_mpo::init2(int a,int b)
{
	cudaMalloc((void**)&mpo2,sizeof(float)*a*b);
}
