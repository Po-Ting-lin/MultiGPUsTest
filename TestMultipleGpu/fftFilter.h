#pragma once
#include <opencv2/opencv.hpp>
#include <cufft.h>
#include "utils.h"

void warmUpGPU();
void runComplexMul(cufftComplex* dst, cufftComplex* src, cufftComplex* kernel, int fftH, int fftW, float c, cudaStream_t& stream);

class Executer {
public:
	Executer() {}
	~Executer() {}

	bool init(int gpuId, int nx, int ny);
	void run(int gpuId, uchar* img);
	void sync();
	bool close(int gpuId);

private:
	int m_nx;
	int m_ny;
	float* m_hImg;
	float* m_hMask;
	cufftComplex* m_hPadImg;
	cufftComplex* m_hPadMask;
	cufftComplex* m_hResultImg;

	float* m_dImg;
	float* m_dMask;
	cufftComplex* m_dPadImg;
	cufftComplex* m_dPadMask;
	cufftComplex* m_dImgSpectrum;
	cufftComplex* m_dMaskSpectrum;
	cufftComplex* m_dMulSpectrum;
	cufftComplex* m_dResultSpectrum;

	cufftHandle m_fftPlanFwd;
	cufftHandle m_fftPlanInv;

	cudaStream_t m_stream;
};