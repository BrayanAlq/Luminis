#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

#include "filesystem_utils.hpp"

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
                 << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        } \
    } while (0)

__global__ void aplicarLUTKernel(const unsigned char* d_imagen_in, 
                                 unsigned char* d_imagen_out, 
                                 const unsigned char* d_lut, 
                                 int totalPix) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalPix) {
        d_imagen_out[i] = d_lut[d_imagen_in[i]];
    }
}

void aplicarLUTCUDA(const std::string& ruta, const std::vector<unsigned char>& lut) {
    cout << "[MASTER-GPU] Aplicando LUT a: " << ruta << "\n";

    cv::Mat imagen_h = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
    if (imagen_h.empty()) {
        cerr << "[MASTER-GPU] Error cargando imagen para GPU: " << ruta << "\n";
        return;
    }

    int rows = imagen_h.rows;
    int cols = imagen_h.cols;
    int totalPix = rows * cols;

    size_t size_imagen = totalPix * sizeof(unsigned char);
    size_t size_lut = 256 * sizeof(unsigned char);

    unsigned char *d_imagen_in = nullptr, *d_imagen_out = nullptr, *d_lut = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_imagen_in, size_imagen));
    CUDA_CHECK(cudaMalloc((void**)&d_imagen_out, size_imagen));
    CUDA_CHECK(cudaMalloc((void**)&d_lut, size_lut));

    CUDA_CHECK(cudaMemcpy(d_imagen_in, imagen_h.data, size_imagen, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lut, lut.data(), size_lut, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPix + threadsPerBlock - 1) / threadsPerBlock;

    aplicarLUTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_imagen_in, d_imagen_out, d_lut, totalPix);
    CUDA_CHECK(cudaGetLastError());

    cv::Mat resultado_h(rows, cols, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(resultado_h.data, d_imagen_out, size_imagen, cudaMemcpyDeviceToHost));

    string ruta_salida = obtenerNombreSalida(ruta);
    cv::imwrite(ruta_salida, resultado_h);

    cout << "[MASTER-GPU] Imagen guardada en: " << ruta_salida << "\n";

    cudaFree(d_imagen_in);
    cudaFree(d_imagen_out);
    cudaFree(d_lut);
}

extern "C" cv::Mat procesarGPU_return_empty() {
    return cv::Mat();
}
