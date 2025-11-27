#include "procesamiento_gpu.hpp"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm> // Para std::min

using namespace std;
namespace fs = std::filesystem;

// --- CONSTANTES ---
const int LUT_SIZE = 256;
const int THREADS_PER_BLOCK = 1024;
const int NUM_STREAMS = 4; // Número de streams que usaremos para paralelizar

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

void aplicarLUTKernel_wrapper(const unsigned char* d_imagen_in, unsigned char* d_imagen_out, const unsigned char* d_lut, int totalPix)
{
    int blocksPerGrid = (totalPix + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    aplicarLUTKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_imagen_in, d_imagen_out, d_lut, totalPix);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// 2. CÓDIGO DEL HOST (MAESTRO - CPU) - Lógica Asíncrona
// -----------------------------------------------------------------------------

// Nueva estructura para agrupar los datos necesarios por stream
struct GpuTask {
    unsigned char *d_imagen_in = nullptr;
    unsigned char *d_imagen_out = nullptr;
    unsigned char *d_lut = nullptr;
    cv::Mat host_output; // Para recibir la imagen procesada de vuelta
    string ruta_salida;
    size_t size_imagen;
    cudaStream_t stream; // Stream dedicado
};


extern "C" void aplicarLUTCUDA_Lote(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data) {
    size_t num_imagenes = rutas.size();
    if (num_imagenes == 0) return;

    cout << "[MASTER-GPU] Procesando lote de " << num_imagenes << " imágenes con " << NUM_STREAMS << " streams.\n";

    // --- 1. Inicialización y Asignación de Streams ---
    vector<GpuTask> tasks(num_imagenes);
    vector<cudaStream_t> streams(NUM_STREAMS);
    for(int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // --- 2. Procesamiento Asíncrono ---
    for (size_t i = 0; i < num_imagenes; ++i) {
        int stream_idx = i % NUM_STREAMS; // Asigna tareas a los streams en round-robin
        
        // Carga de datos para la tarea 'i'
        tasks[i].stream = streams[stream_idx];
        const std::string& ruta = rutas[i];
        
        cv::Mat imagen_h = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
        if (imagen_h.empty()) { cerr << "[MASTER-GPU] Error cargando " << ruta << "\n"; continue; }

        int totalPix = imagen_h.rows * imagen_h.cols;
        tasks[i].size_imagen = totalPix * sizeof(unsigned char);
        tasks[i].ruta_salida = "data/output/ecualizado_" + fs::path(ruta).filename().string();
        
        // Copia local de la LUT del lote
        const unsigned char* lut_ptr = lut_data.data() + (i * LUT_SIZE);
        
        // Asignación de memoria en la GPU (Debe ser síncrona)
        CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_imagen_in, tasks[i].size_imagen));
        CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_imagen_out, tasks[i].size_imagen));
        CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_lut, LUT_SIZE * sizeof(unsigned char)));
        
        // 3. Copia Asíncrona (Host a Device) usando el stream
        // d_imagen_in y d_lut son copiados a la GPU
        CUDA_CHECK(cudaMemcpyAsync(tasks[i].d_imagen_in, imagen_h.data, tasks[i].size_imagen, cudaMemcpyHostToDevice, tasks[i].stream));
        CUDA_CHECK(cudaMemcpyAsync(tasks[i].d_lut, lut_ptr, LUT_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice, tasks[i].stream));
        
        // 4. Lanzamiento del Kernel Asíncrono
        int blocksPerGrid = (totalPix + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        aplicarLUTKernel<<<blocksPerGrid, THREADS_PER_BLOCK, 0, tasks[i].stream>>>(
            tasks[i].d_imagen_in, 
            tasks[i].d_imagen_out, 
            tasks[i].d_lut, 
            totalPix
        );
        CUDA_CHECK(cudaGetLastError()); 

        // 5. Copia Asíncrona de Resultado (Device a Host)
        tasks[i].host_output.create(imagen_h.rows, imagen_h.cols, CV_8UC1);
        CUDA_CHECK(cudaMemcpyAsync(tasks[i].host_output.data, tasks[i].d_imagen_out, tasks[i].size_imagen, cudaMemcpyDeviceToHost, tasks[i].stream));
    }

    // --- 6. Esperar Sincronización y Finalización ---
    // La CPU ahora espera a que cada stream termine su procesamiento
    for(int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // --- 7. Guardar y Limpiar ---
    for (size_t i = 0; i < num_imagenes; ++i) {
        // Guardar la imagen (Operación de I/O de CPU)
        if (!tasks[i].host_output.empty()) {
            cv::imwrite(tasks[i].ruta_salida, tasks[i].host_output);
            // cout << "[MASTER-GPU] Imagen guardada en: " << tasks[i].ruta_salida << "\n";
        }
        
        // Liberar memoria de la GPU
        CUDA_CHECK(cudaFree(tasks[i].d_imagen_in));
        CUDA_CHECK(cudaFree(tasks[i].d_imagen_out));
        CUDA_CHECK(cudaFree(tasks[i].d_lut));
    }
    
    // Destruir los streams
    for(int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    
    cout << "[MASTER-GPU] Lote procesado asíncronamente y guardado.\n";
}

// Placeholder
extern "C" cv::Mat procesarGPU_return_empty() {
    return cv::Mat();
}