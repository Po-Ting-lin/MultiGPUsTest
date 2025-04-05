#include <iostream>
#include "fftFilter.h"

void run(Executer* obj, int gpuId, uchar* data, int times) {
    for (int i = 0; i < times; i++) {
        printf("run: Device: %d Iteration: %d Start\n", gpuId, i);
        obj->run(gpuId, data);
        printf("run: Device: %d Iteration: %d Done!\n", gpuId, i);
    }
    obj->sync();
}

int main() {
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