#include "procesamiento_gpu.hpp"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm> // Para std::min
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>

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

#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
// --- IMPLEMENTACIÓN DEL MEMORY POOL CUDA ---
CudaMemoryPool g_memory_pool;

void CudaMemoryPool::initialize(size_t max_img_size, size_t max_lut_sz) {
    if (initialized) {
        cleanup();
    }
    
    max_image_size = max_img_size;
    max_lut_size = max_lut_sz;
    
    // Asignar memoria para el pool
    CUDA_CHECK(cudaMalloc(&image_buffer, max_image_size));
    CUDA_CHECK(cudaMalloc(&output_buffer, max_image_size));
    CUDA_CHECK(cudaMalloc(&lut_buffer, max_lut_size));
    
    initialized = true;
    cout << "[CUDA-POOL] Memory pool initialized. Max image size: " << max_image_size << " bytes" << endl;
}

unsigned char* CudaMemoryPool::getImageBuffer(size_t requested_size) {
    if (!initialized || requested_size > max_image_size) {
        cerr << "[CUDA-POOL] Error: Pool not initialized or size too large" << endl;
        return nullptr;
    }
    return image_buffer;
}

unsigned char* CudaMemoryPool::getOutputBuffer(size_t requested_size) {
    if (!initialized || requested_size > max_image_size) {
        cerr << "[CUDA-POOL] Error: Pool not initialized or size too large" << endl;
        return nullptr;
    }
    return output_buffer;
}

unsigned char* CudaMemoryPool::getLUTBuffer(size_t requested_size) {
    if (!initialized || requested_size > max_lut_size) {
        cerr << "[CUDA-POOL] Error: Pool not initialized or LUT size too large" << endl;
        return nullptr;
    }
    return lut_buffer;
}

void CudaMemoryPool::releaseImageBuffer() {
    // En esta implementación simple, no liberamos individualmente
    // El pool se libera completamente en cleanup()
}

void CudaMemoryPool::releaseOutputBuffer() {
    // En esta implementación simple, no liberamos individualmente
    // El pool se libera completamente en cleanup()
}

void CudaMemoryPool::releaseLutBuffer() {
    // En esta implementación simple, no liberamos individualmente
    // El pool se libera completamente en cleanup()
}

void CudaMemoryPool::cleanup() {
    if (initialized) {
        CUDA_CHECK(cudaFree(image_buffer));
        CUDA_CHECK(cudaFree(output_buffer));
        CUDA_CHECK(cudaFree(lut_buffer));
        initialized = false;
        cout << "[CUDA-POOL] Memory pool cleaned up" << endl;
    }
}

#ifdef ENABLE_ASYNC_IO_OPT
// Instancia global del gestor I/O
AsyncIOManager g_async_io(50); // Máximo 50 tareas en cola

// Implementación de AsyncIOManager
void AsyncIOManager::ioWorkerThread() {
    cout << "[ASYNC-IO] Thread de I/O iniciado" << endl;
    
    while (running.load() || !save_queue.empty()) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        // Esperar por trabajo o señal de terminación
        queue_cv.wait(lock, [this] { return !save_queue.empty() || !running.load(); });
        
        if (!save_queue.empty()) {
            SaveTask task = save_queue.front();
            save_queue.pop();
            lock.unlock();
            
            // Procesar tarea de guardado
            try {
                bool success = cv::imwrite(task.output_path, task.image);
                if (success) {
                    cout << "[ASYNC-IO] Imagen guardada: " << task.output_path << " (task " << task.task_id << ")" << endl;
                } else {
                    cerr << "[ASYNC-IO] Error guardando: " << task.output_path << " (task " << task.task_id << ")" << endl;
                }
            } catch (const cv::Exception& e) {
                cerr << "[ASYNC-IO] OpenCV exception guardando " << task.output_path << ": " << e.what() << endl;
            } catch (const std::exception& e) {
                cerr << "[ASYNC-IO] Exception guardando " << task.output_path << ": " << e.what() << endl;
            }
        } else {
            lock.unlock();
        }
    }
    
    cout << "[ASYNC-IO] Thread de I/O finalizado" << endl;
}

void AsyncIOManager::start() {
    if (running.load()) {
        return; // Ya está corriendo
    }
    
    running = true;
    io_thread = std::thread(&AsyncIOManager::ioWorkerThread, this);
}

bool AsyncIOManager::enqueueSave(const cv::Mat& img, const std::string& path) {
    if (!running.load()) {
        cerr << "[ASYNC-IO] Error: AsyncIOManager no está iniciado" << endl;
        return false;
    }
    
    std::unique_lock<std::mutex> lock(queue_mutex);
    
    // Verificar límite de cola para evitar overflow de memoria
    if (save_queue.size() >= max_queue_size) {
        cerr << "[ASYNC-IO] Advertencia: Cola de I/O llena (" << save_queue.size() << "/" << max_queue_size << ")" << endl;
        return false;
    }
    
    int task_id = next_task_id.fetch_add(1);
    save_queue.emplace(img, path, task_id);
    lock.unlock();
    
    queue_cv.notify_one();
    return true;
}

void AsyncIOManager::waitForCompletion() {
    if (!running.load()) {
        return;
    }
    
    // Esperar a que la cola se vacíe
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [this] { return save_queue.empty(); });
}

void AsyncIOManager::stop() {
    if (!running.load()) {
        return;
    }
    
    running = false;
    queue_cv.notify_all();
    
    if (io_thread.joinable()) {
        io_thread.join();
    }
    
    cout << "[ASYNC-IO] AsyncIOManager detenido" << endl;
}

size_t AsyncIOManager::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return save_queue.size();
}

// Función modificada para usar I/O asíncrono
void aplicarLUTCUDA_Lote_AsyncIO(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data);
#endif
#endif

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


void aplicarLUTCUDA_Lote(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data) {
#ifdef ENABLE_ASYNC_IO_OPT
    cout << "[MASTER-GPU] Procesando lote con I/O asíncrono: " << rutas.size() << " imágenes.\n";
    aplicarLUTCUDA_Lote_AsyncIO(rutas, lut_data);
#else
    cout << "[MASTER-GPU] Procesando lote de " << rutas.size() << " imágenes con " << NUM_STREAMS << " streams.\n";

#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
    // --- 0. Inicialización del Memory Pool ---
    size_t max_image_size = 0;
    for (const auto& ruta : rutas) {
        cv::Mat temp_img = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
        if (!temp_img.empty()) {
            size_t img_size = temp_img.rows * temp_img.cols * sizeof(unsigned char);
            max_image_size = max(max_image_size, img_size);
        }
    }
    
    // Inicializar pool con tamaño máximo encontrado
    g_memory_pool.initialize(max_image_size, LUT_SIZE * sizeof(unsigned char));
    cout << "[MASTER-GPU] Memory pool initialized for max image size: " << max_image_size << " bytes" << endl;
#endif

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
        
// Asignación de memoria en la GPU
#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
        // Usar Memory Pool para obtener buffers pre-asignados
        tasks[i].d_imagen_in = g_memory_pool.getImageBuffer(tasks[i].size_imagen);
        tasks[i].d_imagen_out = g_memory_pool.getOutputBuffer(tasks[i].size_imagen);
        tasks[i].d_lut = g_memory_pool.getLUTBuffer(LUT_SIZE * sizeof(unsigned char));
#else
        // Asignación tradicional (baseline)
        CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_imagen_in, tasks[i].size_imagen));
        CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_imagen_out, tasks[i].size_imagen));
        CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_lut, LUT_SIZE * sizeof(unsigned char)));
#endif
        
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
#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
        // Con Memory Pool, no liberamos individualmente
        g_memory_pool.releaseImageBuffer();
        g_memory_pool.releaseOutputBuffer();
        g_memory_pool.releaseLutBuffer();
#else
        // Liberación tradicional (baseline)
        CUDA_CHECK(cudaFree(tasks[i].d_imagen_in));
        CUDA_CHECK(cudaFree(tasks[i].d_imagen_out));
        CUDA_CHECK(cudaFree(tasks[i].d_lut));
#endif
    }
    
// Destruir los streams
    for(int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    
#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
    // Limpiar el Memory Pool
    g_memory_pool.cleanup();
#endif
    
    cout << "[MASTER-GPU] Lote procesado asíncronamente y guardado.\n";
#endif
}

#ifdef ENABLE_ASYNC_IO_OPT
// -----------------------------------------------------------------------------
// Implementación con Pipeline CUDA + I/O Asíncrono
// -----------------------------------------------------------------------------
void aplicarLUTCUDA_Lote_AsyncIO(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data) {
    size_t num_imagenes = rutas.size();
    if (num_imagenes == 0) return;

    // Iniciar el gestor de I/O asíncrono
    g_async_io.start();
    cout << "[ASYNC-IO] Gestor de I/O iniciado para " << num_imagenes << " imágenes\n";

#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
    // --- 0. Inicialización del Memory Pool ---
    size_t max_image_size = 0;
    for (const auto& ruta : rutas) {
        cv::Mat temp_img = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
        if (!temp_img.empty()) {
            size_t img_size = temp_img.rows * temp_img.cols * sizeof(unsigned char);
            max_image_size = max(max_image_size, img_size);
        }
    }
    
    // Inicializar pool con tamaño máximo encontrado
    g_memory_pool.initialize(max_image_size, LUT_SIZE * sizeof(unsigned char));
    cout << "[ASYNC-IO] Memory pool initialized for max image size: " << max_image_size << " bytes" << endl;
#endif

    // --- 1. Inicialización y Asignación de Streams ---
    vector<GpuTask> tasks(num_imagenes);
    vector<cudaStream_t> streams(NUM_STREAMS);
    for(int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // --- 2. Procesamiento Asíncrono con Pipeline ---
    const int BATCH_SIZE = 8; // Procesar en lotes más pequeños para mejor pipeline
    size_t processed_count = 0;
    
    while (processed_count < num_imagenes) {
        size_t batch_end = min(processed_count + BATCH_SIZE, num_imagenes);
        
        cout << "[ASYNC-IO] Procesando batch " << (processed_count/BATCH_SIZE + 1) 
             << " (imágenes " << processed_count << "-" << (batch_end-1) << ")\n";
        
        // 2.1. Procesar lote actual en GPU
        for (size_t i = processed_count; i < batch_end; ++i) {
            int stream_idx = i % NUM_STREAMS;
            
            // Carga de datos para la tarea 'i'
            tasks[i].stream = streams[stream_idx];
            const std::string& ruta = rutas[i];
            
            cv::Mat imagen_h = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
            if (imagen_h.empty()) { 
                cerr << "[ASYNC-IO] Error cargando " << ruta << "\n"; 
                continue; 
            }

            int totalPix = imagen_h.rows * imagen_h.cols;
            tasks[i].size_imagen = totalPix * sizeof(unsigned char);
            tasks[i].ruta_salida = "data/output/ecualizado_" + fs::path(ruta).filename().string();
            
            // Copia local de la LUT del lote
            const unsigned char* lut_ptr = lut_data.data() + (i * LUT_SIZE);
            
            // Asignación de memoria en la GPU
#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
            // Usar Memory Pool para obtener buffers pre-asignados
            tasks[i].d_imagen_in = g_memory_pool.getImageBuffer(tasks[i].size_imagen);
            tasks[i].d_imagen_out = g_memory_pool.getOutputBuffer(tasks[i].size_imagen);
            tasks[i].d_lut = g_memory_pool.getLUTBuffer(LUT_SIZE * sizeof(unsigned char));
#else
            // Asignación tradicional (baseline)
            CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_imagen_in, tasks[i].size_imagen));
            CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_imagen_out, tasks[i].size_imagen));
CUDA_CHECK(cudaMalloc((void**)&tasks[i].d_lut, LUT_SIZE * sizeof(unsigned char));
#endif
            
            // 2.2. Copia Asíncrona (Host a Device) usando el stream
            CUDA_CHECK(cudaMemcpyAsync(tasks[i].d_imagen_in, imagen_h.data, tasks[i].size_imagen, cudaMemcpyHostToDevice, tasks[i].stream));
            CUDA_CHECK(cudaMemcpyAsync(tasks[i].d_lut, lut_ptr, LUT_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice, tasks[i].stream));
            
            // 2.3. Lanzamiento del Kernel Asíncrono
            int blocksPerGrid = (totalPix + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            
            aplicarLUTKernel<<<blocksPerGrid, THREADS_PER_BLOCK, 0, tasks[i].stream>>>(
                tasks[i].d_imagen_in, 
                tasks[i].d_imagen_out, 
                tasks[i].d_lut, 
                totalPix
            );
            CUDA_CHECK(cudaGetLastError()); 

            // 2.4. Copia Asíncrona de Resultado (Device a Host)
            tasks[i].host_output.create(imagen_h.rows, imagen_h.cols, CV_8UC1);
            CUDA_CHECK(cudaMemcpyAsync(tasks[i].host_output.data, tasks[i].d_imagen_out, tasks[i].size_imagen, cudaMemcpyDeviceToHost, tasks[i].stream));
        }
        
        // 2.5. Esperar a que complete el batch actual
        for(int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        
        // 2.6. Encolar tareas de I/O asíncrono para el batch procesado
        for (size_t i = processed_count; i < batch_end; ++i) {
            if (!tasks[i].host_output.empty()) {
                bool enqueued = g_async_io.enqueueSave(tasks[i].host_output, tasks[i].ruta_salida);
                if (!enqueued) {
                    // Fallback a I/O síncrono si la cola está llena
                    cv::imwrite(tasks[i].ruta_salida, tasks[i].host_output);
                    cout << "[ASYNC-IO] Fallback síncrono para: " << tasks[i].ruta_salida << endl;
                }
            }
            
            // Liberar memoria de la GPU
#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
            // Con Memory Pool, no liberamos individualmente
            g_memory_pool.releaseImageBuffer();
            g_memory_pool.releaseOutputBuffer();
            g_memory_pool.releaseLutBuffer();
#else
            // Liberación tradicional (baseline)
            CUDA_CHECK(cudaFree(tasks[i].d_imagen_in));
            CUDA_CHECK(cudaFree(tasks[i].d_imagen_out));
            CUDA_CHECK(cudaFree(tasks[i].d_lut));
#endif
        }
        
        processed_count = batch_end;
        
        // 2.7. Mostrar estado de la cola de I/O
        size_t queue_size = g_async_io.getQueueSize();
        if (queue_size > 0) {
            cout << "[ASYNC-IO] Cola de I/O: " << queue_size << " tareas pendientes\n";
        }
    }
    
    // --- 3. Destruir los streams ---
    for(int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    
#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
    // Limpiar el Memory Pool
    g_memory_pool.cleanup();
#endif
    
    // --- 4. Esperar a que todas las tareas de I/O completen ---
    cout << "[ASYNC-IO] Esperando finalización de tareas de I/O...\n";
    g_async_io.waitForCompletion();
    
    // --- 5. Detener el gestor de I/O asíncrono ---
    g_async_io.stop();
    
    cout << "[ASYNC-IO] Lote procesado con pipeline CUDA + I/O asíncrono.\n";
}
#endif

// Placeholder
extern "C" cv::Mat procesarGPU_return_empty() {
    return cv::Mat();
}
