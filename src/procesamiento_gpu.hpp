#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>


// Declare the kernel launcher as an extern "C" function
extern "C" void aplicarLUTCUDA_Lote(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data);
extern "C" cv::Mat procesarGPU_return_empty();

// We need to declare the kernel wrapper to be able to call it from the test
void aplicarLUTKernel_wrapper(const unsigned char* d_imagen_in, unsigned char* d_imagen_out, const unsigned char* d_lut, int totalPix);

#ifdef ENABLE_CUDA_MEMORY_POOL_OPT
// Memory Pool para gestión eficiente de memoria CUDA
struct CudaMemoryPool {
    unsigned char* image_buffer;
    unsigned char* output_buffer;
    unsigned char* lut_buffer;
    size_t max_image_size;
    size_t max_lut_size;
    bool initialized;
    
    // Inicialización del pool
    void initialize(size_t max_img_size, size_t max_lut_sz = 256);
    
    // Obtener buffers del pool
    unsigned char* getImageBuffer(size_t requested_size);
    unsigned char* getOutputBuffer(size_t requested_size);
    unsigned char* getLUTBuffer(size_t requested_size);
    
    // Liberar buffers (devolver al pool)
    void releaseImageBuffer();
    void releaseOutputBuffer();
    void releaseLutBuffer();
    
    // Limpieza completa
    void cleanup();
};

// Instancia global del memory pool
extern CudaMemoryPool g_memory_pool;

#ifdef ENABLE_ASYNC_IO_OPT
// Tarea de guardado asíncrono
struct SaveTask {
    cv::Mat image;
    std::string output_path;
    int task_id;
    
    SaveTask(const cv::Mat& img, const std::string& path, int id) 
        : image(img.clone()), output_path(path), task_id(id) {}
};

// Gestor de I/O asíncrono
class AsyncIOManager {
private:
    std::queue<SaveTask> save_queue;
	mutable std::mutex queue_mutex;    
    std::condition_variable queue_cv;
    std::thread io_thread;
    std::atomic<bool> running{false};
    std::atomic<int> next_task_id{0};
    int max_queue_size;
    
    // Thread worker para I/O
    void ioWorkerThread();
    
public:
    AsyncIOManager(int max_size = 50) : max_queue_size(max_size) {}
    
    // Iniciar el thread de I/O
    void start();
    
    // Encolar tarea de guardado (retorna inmediatamente)
    bool enqueueSave(const cv::Mat& img, const std::string& path);
    
    // Esperar a que todas las tareas completen
    void waitForCompletion();
    
    // Detener el thread
    void stop();
    
    // Obtener estadísticas
    size_t getQueueSize() const;
    bool isRunning() const { return running.load(); }
};

// Instancia global del gestor I/O
extern AsyncIOManager g_async_io;
#endif
#endif
