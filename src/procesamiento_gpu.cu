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
// 1. CÓDIGO DEL DEVICE (KERNEL CUDA - 1D)
// -----------------------------------------------------------------------------
__global__ void aplicarLUTKernel(const unsigned char* d_imagen_in, 
                                 unsigned char* d_imagen_out, 
                                 const unsigned char* d_lut, 
                                 int totalPix) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < totalPix) {
        d_imagen_out[i] = d_lut[d_imagen_in[i]];
    }
}

// -----------------------------------------------------------------------------
// 2. CÓDIGO DEL HOST (MAESTRO - CPU)
// -----------------------------------------------------------------------------
// Función auxiliar que aplica CUDA a una sola imagen (llamada por la función de lote)
extern "C" void aplicarLUTCUDA_Individual(const std::string& ruta, const std::vector<unsigned char>& lut) {
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
    
    // 2. Asignación y Copia de memoria
    CUDA_CHECK(cudaMalloc((void**)&d_imagen_in, size_imagen));
    CUDA_CHECK(cudaMalloc((void**)&d_imagen_out, size_imagen));
    CUDA_CHECK(cudaMalloc((void**)&d_lut, size_lut));
    
    CUDA_CHECK(cudaMemcpy(d_imagen_in, imagen_h.data, size_imagen, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lut, lut.data(), size_lut, cudaMemcpyHostToDevice));
    
    // 3. Configuración y Lanzamiento del Kernel 1D
    int threadsPerBlock = 1024; 
    int blocksPerGrid = (totalPix + threadsPerBlock - 1) / threadsPerBlock;
    
    aplicarLUTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_imagen_in, d_imagen_out, d_lut, totalPix);
    CUDA_CHECK(cudaGetLastError()); 

    // 4. Copiar resultado
    cv::Mat resultado_h(rows, cols, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(resultado_h.data, d_imagen_out, size_imagen, cudaMemcpyDeviceToHost));
    
    // 5. Guardar la imagen
    string ruta_salida = "data/output/ecualizado_" + fs::path(ruta).filename().string();
    cv::imwrite(ruta_salida, resultado_h);
    
    cout << "[MASTER-GPU] Imagen guardada en: " << ruta_salida << "\n";

    // 6. Liberar memoria
    CUDA_CHECK(cudaFree(d_imagen_in));
    CUDA_CHECK(cudaFree(d_imagen_out));
    CUDA_CHECK(cudaFree(d_lut));
}

// -----------------------------------------------------------------------------
// Función de Lote para el Maestro (Llamada por gestor_distribucion.cpp)
// -----------------------------------------------------------------------------
extern "C" void aplicarLUTCUDA_Lote(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data) {
    const int LUT_SIZE = 256;
    size_t num_imagenes = rutas.size();

    if (num_imagenes == 0) return;

    for (size_t i = 0; i < num_imagenes; ++i) {
        // Extraer la LUT para la imagen actual del array contiguo
        std::vector<unsigned char> lut_individual(
            lut_data.begin() + (i * LUT_SIZE),
            lut_data.begin() + ((i + 1) * LUT_SIZE)
        );

        const std::string& ruta = rutas[i];

        // Aplicar CUDA individualmente
        aplicarLUTCUDA_Individual(ruta, lut_individual);
    }
}

// Placeholder
extern "C" cv::Mat procesarGPU_return_empty() {
    return cv::Mat();
}