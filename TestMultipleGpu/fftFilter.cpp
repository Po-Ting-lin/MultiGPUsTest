#include "fftFilter.h"

bool Executer::init(int gpuId, int nx, int ny) {
	Check(cudaSetDevice(gpuId));
	m_nx = cv::getOptimalDFTSize(nx);
	m_ny = cv::getOptimalDFTSize(ny);

	const float c = 1.0f / (m_nx * m_ny);
	Check(cudaMallocHost(&m_hMask, (size_t)m_nx * m_ny * sizeof(float)));
	Check(cudaMallocHost(&m_hPadImg, (size_t)m_nx * m_ny * sizeof(cufftComplex)));
	Check(cudaMallocHost(&m_hPadMask, (size_t)m_nx * m_ny * sizeof(cufftComplex)));
	Check(cudaMallocHost(&m_hResultImg, (size_t)m_nx * m_ny * sizeof(cufftComplex)));
	Check(cudaMalloc(&m_dImg, (size_t)m_nx * m_ny * sizeof(float)));
	Check(cudaMalloc(&m_dMask, (size_t)m_nx * m_ny * sizeof(float)));
	Check(cudaMalloc(&m_dPadImg, (size_t)m_nx * m_ny * sizeof(cufftComplex)));
	Check(cudaMalloc(&m_dPadMask, (size_t)m_nx * m_ny * sizeof(cufftComplex)));
	Check(cudaMalloc(&m_dImgSpectrum, (size_t)m_nx * m_ny * sizeof(cufftComplex)));
	Check(cudaMalloc(&m_dMaskSpectrum, (size_t)m_nx * m_ny * sizeof(cufftComplex)));
	Check(cudaMalloc(&m_dMulSpectrum, (size_t)m_nx * m_ny * sizeof(cufftComplex)));
	Check(cudaMalloc(&m_dResultSpectrum, (size_t)m_nx * m_ny * sizeof(cufftComplex)));

	Check(cudaStreamCreate(&m_stream));
	Check(cufftPlan2d(&m_fftPlanFwd, m_ny, m_nx, CUFFT_C2C));
	Check(cufftPlan2d(&m_fftPlanInv, m_ny, m_nx, CUFFT_C2C));
	Check(cufftSetStream(m_fftPlanFwd, m_stream));
	Check(cufftSetStream(m_fftPlanInv, m_stream));
	return true;
}

void Executer::run(int gpuId, uchar* img) {
	Check(cudaSetDevice(gpuId));
	const float c = 1.0f / (m_nx * m_ny);
	Check(cudaMemcpyAsync(m_dPadImg, m_hPadImg, (size_t)m_nx * m_ny * sizeof(cufftComplex), cudaMemcpyHostToDevice, m_stream));
	//Check(cudaMemcpyAsync(m_dMaskSpectrum, m_hPadMask, (size_t)m_nx * m_ny * sizeof(cufftComplex), cudaMemcpyHostToDevice, m_stream));

	if (gpuId == 0) {
		for (int i = 0; i < 10; i++) {
			Check(cufftExecC2C(m_fftPlanFwd, m_dPadImg, m_dImgSpectrum, CUFFT_FORWARD));
			runComplexMul(m_dMulSpectrum, m_dImgSpectrum, m_dMaskSpectrum, m_ny, m_nx, c, m_stream);
			Check(cufftExecC2C(m_fftPlanInv, m_dMulSpectrum, m_dResultSpectrum, CUFFT_INVERSE));
		}
	}
	else {
		Check(cufftExecC2C(m_fftPlanFwd, m_dPadImg, m_dImgSpectrum, CUFFT_FORWARD));
		runComplexMul(m_dMulSpectrum, m_dImgSpectrum, m_dMaskSpectrum, m_ny, m_nx, c, m_stream);
		Check(cufftExecC2C(m_fftPlanInv, m_dMulSpectrum, m_dResultSpectrum, CUFFT_INVERSE));
	}
	Check(cudaMemcpyAsync(m_hResultImg, m_dResultSpectrum, (size_t)m_nx * m_ny * sizeof(cufftComplex), cudaMemcpyDeviceToHost, m_stream));
}

void Executer::sync() {
	Check(cudaStreamSynchronize(m_stream));
}

bool Executer::close(int gpuId) {
	Check(cudaSetDevice(gpuId));
	Check(cufftDestroy(m_fftPlanFwd));
	Check(cufftDestroy(m_fftPlanInv));
	Check(cudaFree(m_dImg));
	Check(cudaFree(m_dMask));
	Check(cudaFree(m_dPadImg));
	Check(cudaFree(m_dPadMask));
	Check(cudaFree(m_dImgSpectrum));
	Check(cudaFree(m_dMaskSpectrum));
	Check(cudaFree(m_dMulSpectrum));
	Check(cudaFree(m_dResultSpectrum));
	Check(cudaFreeHost(m_hMask));
	Check(cudaFreeHost(m_hPadImg));
	Check(cudaFreeHost(m_hPadMask));
	Check(cudaFreeHost(m_hResultImg));
}