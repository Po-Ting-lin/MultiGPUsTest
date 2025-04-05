#pragma once
#include <string>
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaHelper.h"
#define ROUND(a) (uchar)(a + 0.55f)
#define WARP_SIZE 32

typedef unsigned long long uintll;
typedef unsigned char uchar;

inline int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline int iAlignUp(int a, int b) {
    return (a % b != 0) ? (a - a % b + b) : a;
}

static inline std::chrono::system_clock::time_point getTime() {
    return std::chrono::system_clock::now();
}

static inline double getDurationMs(std::chrono::system_clock::time_point t1, std::chrono::system_clock::time_point t2) {
    /* Getting number of milliseconds as an integer. */
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    /* Getting number of milliseconds as a double. */
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    return ms_double.count();
}