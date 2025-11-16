#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

// --- FUNCIÓN DE UTILIDAD: MANEJO DE ERRORES CUDA ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "[CUDA ERROR] " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        } \
    } while (0)

// -----------------------------------------------------------------------------
// 1. CÓDIGO DEL DEVICE (KERNEL CUDA - 1D EFICIENTE)
// totalPix es el número total de elementos (filas * columnas)
// -----------------------------------------------------------------------------
__global__ void aplicarLUTKernel(const unsigned char* d_imagen_in, 
                                 unsigned char* d_imagen_out, 
                                 const unsigned char* d_lut, 
                                 int totalPix) {
    
    // Calcula el índice global 1D
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < totalPix) {
        // Aplica la LUT: el valor original es el índice en la tabla
        d_imagen_out[i] = d_lut[d_imagen_in[i]];
    }
}

// -----------------------------------------------------------------------------
// 2. CÓDIGO DEL HOST (MAESTRO - CPU)
// La función ahora calcula el totalPix y lanza el kernel 1D
// -----------------------------------------------------------------------------
extern "C" void aplicarLUTCUDA(const std::string& ruta, const std::vector<unsigned char>& lut) {
    cout << "[MASTER-GPU] Aplicando LUT a: " << ruta << "\n";
    
    // 1. Cargar la imagen y obtener tamaños
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
    
    // 2. Asignación y Copia de memoria en la GPU
    CUDA_CHECK(cudaMalloc((void**)&d_imagen_in, size_imagen));
    CUDA_CHECK(cudaMalloc((void**)&d_imagen_out, size_imagen));
    CUDA_CHECK(cudaMalloc((void**)&d_lut, size_lut));
    
    if (!d_imagen_in || !d_imagen_out || !d_lut) {
        // ... manejo de error ...
        cudaFree(d_imagen_in); cudaFree(d_imagen_out); cudaFree(d_lut);
        return;
    }

    CUDA_CHECK(cudaMemcpy(d_imagen_in, imagen_h.data, size_imagen, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lut, lut.data(), size_lut, cudaMemcpyHostToDevice));
    
    // 3. Configuración y Lanzamiento del Kernel 1D
    int threadsPerBlock = 256; // Un tamaño común para hilos por bloque
    int blocksPerGrid = (totalPix + threadsPerBlock - 1) / threadsPerBlock;
    
    // Lanzamiento del Kernel 1D
    aplicarLUTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_imagen_in, d_imagen_out, d_lut, totalPix);
    CUDA_CHECK(cudaGetLastError()); // Chequeo de errores de lanzamiento

    // 4. Copiar resultado del Device (GPU) al Host (CPU)
    cv::Mat resultado_h(rows, cols, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(resultado_h.data, d_imagen_out, size_imagen, cudaMemcpyDeviceToHost));
    
    // 5. Guardar la imagen ecualizada
    string ruta_salida = "data/output/ecualizado_" + fs::path(ruta).filename().string();
    cv::imwrite(ruta_salida, resultado_h);
    
    cout << "[MASTER-GPU] Imagen guardada en: " << ruta_salida << "\n";

    // 6. Liberar memoria de la GPU
    CUDA_CHECK(cudaFree(d_imagen_in));
    CUDA_CHECK(cudaFree(d_imagen_out));
    CUDA_CHECK(cudaFree(d_lut));
}

// Placeholder (se mantiene por si acaso es usado en alguna otra parte)
extern "C" cv::Mat procesarGPU_return_empty() {
    return cv::Mat();
}