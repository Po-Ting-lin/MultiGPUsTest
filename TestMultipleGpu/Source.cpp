#include <iostream>
#include "fftFilter.h"

void queryDevice() {
	int deviceCount = 0;
	Check(cudaGetDeviceCount(&deviceCount));
	std::cout << "Current Device Count: " << deviceCount << std::endl;

	for (int deviceIdx = 0; deviceIdx < deviceCount; deviceIdx++) {
		cudaDeviceProp myProperty;
		Check(cudaGetDeviceProperties(&myProperty, deviceIdx));

		std::cout << "------------------------------------------------------------------" << std::endl;
		printf("Device %d: %s\n", deviceIdx, myProperty.name);
		printf("  Number of multiprocessors:                     %d\n",
			myProperty.multiProcessorCount);
		printf("  clock rate :                     %4.2f MHz\n",
			myProperty.clockRate / 1000.0);
		printf("  Compute capability       :                     %d.%d\n",
			myProperty.major, myProperty.minor);
		printf("  Total amount of global memory:                 %4.2f MB\n",
			myProperty.totalGlobalMem / 1024.0 / 1024.0);
		printf("  Total amount of constant memory:               %4.2f KB\n",
			myProperty.totalConstMem / 1024.0);
		printf("  Total amount of shared memory per block:       %4.2f KB\n",
			myProperty.sharedMemPerBlock / 1024.0);
		printf("  Total amount of shared memory per MP:          %4.2f KB\n",
			myProperty.sharedMemPerMultiprocessor / 1024.0);
		printf("  Total number of registers available per block: %d\n",
			myProperty.regsPerBlock);
		printf("  Total number of registers available per multiprocessor: %d\n",
			myProperty.regsPerMultiprocessor);
		printf("  Warp size:                                     %d\n",
			myProperty.warpSize);
		printf("  Maximum number of threads per block:           %d\n",
			myProperty.maxThreadsPerBlock);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
			myProperty.maxThreadsPerMultiProcessor);
		printf("  Maximum number of warps per multiprocessor:    %d\n",
			myProperty.maxThreadsPerMultiProcessor / 32);
		printf("  Maximum Grid size                         :    (%d,%d,%d)\n",
			myProperty.maxGridSize[0], myProperty.maxGridSize[1], myProperty.maxGridSize[2]);
		printf("  Maximum block dimension                   :    (%d,%d,%d)\n",
			myProperty.maxThreadsDim[0], myProperty.maxThreadsDim[1], myProperty.maxThreadsDim[2]);
		std::cout << "------------------------------------------------------------------" << std::endl;
	}
}

void topoQuery() {
	std::cout << "------------------------------------------------------------------" << std::endl;
	int deviceCount = 0;
	Check(cudaGetDeviceCount(&deviceCount));

	// Enumerates Device <-> Device links
	for (int device1 = 0; device1 < deviceCount; device1++) {
		for (int device2 = 0; device2 < deviceCount; device2++) {
			if (device1 == device2) continue;

			int perfRank = 0;
			int atomicSupported = 0;
			int accessSupported = 0;

			Check(cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2));
			Check(cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, device1, device2));
			Check(cudaDeviceGetP2PAttribute(&atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1, device2));

			if (accessSupported) {
				std::cout << "GPU" << device1 << " <-> GPU" << device2 << ":" << std::endl;
				std::cout << "  * Atomic Supported: " << (atomicSupported ? "yes" : "no") << std::endl;
				std::cout << "  * Perf Rank: " << perfRank << std::endl;
			}
		}
	}

	// Enumerates Device <-> Host links
	for (int device = 0; device < deviceCount; device++) {
		int atomicSupported = 0;
		Check(cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, device));
		std::cout << "GPU" << device << " <-> CPU:" << std::endl;
		std::cout << "  * Atomic Supported: " << (atomicSupported ? "yes" : "no") << std::endl;
	}
	std::cout << "------------------------------------------------------------------" << std::endl;
}

void run(Executer* obj, int gpuId, uchar* data, int times) {
    for (int i = 0; i < times; i++) {
        printf("run: Device: %d Iteration: %d Start\n", gpuId, i);
        obj->run(gpuId, data);
        printf("run: Device: %d Iteration: %d Done!\n", gpuId, i);
    }
    obj->sync();
}

int main() {
	queryDevice();
	topoQuery();
	const int nx = 2048;
	const int ny = 2048;
    const int times = 50;
	uchar* data = new uchar[nx * ny];
	float* results = new float[nx * ny];
    Executer* obj1 = new Executer();
    Executer* obj2 = new Executer();

    obj1->init(0, nx, ny);
    obj2->init(1, 1024, 1024);

    std::thread t1(run, obj1, 0, data, times);
    std::thread t2(run, obj2, 1, data, times);
    t1.join();
    t2.join();

    obj1->close(0);
    obj2->close(1);

    delete obj1;
    delete obj2;
	delete[] data;
	delete[] results;
	return 0;
}