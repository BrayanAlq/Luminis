#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "procesamiento_gpu.hpp"

// Simple macro to handle CUDA errors in the test
#define CUDA_TEST_CHECK(call) \
    ASSERT_EQ(cudaSuccess, call)

TEST(ProcesamientoGpuTest, AplicarLUTKernel) {
    // 1. Test data
    const int image_size = 256;
    unsigned char host_image_in[image_size];
    unsigned char host_lut[256];
    unsigned char host_image_out[image_size];
    unsigned char expected_image_out[image_size];

    for (int i = 0; i < 256; ++i) {
        host_lut[i] = 255 - i;
    }

    for (int i = 0; i < image_size; ++i) {
        host_image_in[i] = i;
        expected_image_out[i] = host_lut[i];
    }

    // 2. Allocate memory on the device
    unsigned char* device_image_in;
    unsigned char* device_lut;
    unsigned char* device_image_out;

    CUDA_TEST_CHECK(cudaMalloc((void**)&device_image_in, image_size * sizeof(unsigned char)));
    CUDA_TEST_CHECK(cudaMalloc((void**)&device_lut, 256 * sizeof(unsigned char)));
    CUDA_TEST_CHECK(cudaMalloc((void**)&device_image_out, image_size * sizeof(unsigned char)));

    // 3. Copy data from host to device
    CUDA_TEST_CHECK(cudaMemcpy(device_image_in, host_image_in, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(device_lut, host_lut, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // 4. Launch the kernel
    aplicarLUTKernel_wrapper(device_image_in, device_image_out, device_lut, image_size);

    // 5. Copy results back from device to host
    CUDA_TEST_CHECK(cudaMemcpy(host_image_out, device_image_out, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // 6. Compare results
    for (int i = 0; i < image_size; ++i) {
        ASSERT_EQ(host_image_out[i], expected_image_out[i]);
    }

    // 7. Free device memory
    CUDA_TEST_CHECK(cudaFree(device_image_in));
    CUDA_TEST_CHECK(cudaFree(device_lut));
    CUDA_TEST_CHECK(cudaFree(device_image_out));
}
