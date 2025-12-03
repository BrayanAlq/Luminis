#pragma once

#include <vector>
#include <string>
#include <queue>
#include <mutex>

void ejecutarMaestro(int world_size);
void ejecutarEsclavo(int rank);

#ifdef ENABLE_WORK_STEALING_OPT
// Nuevos tags MPI para work stealing
const int TAG_REQUEST_WORK = 30;
const int TAG_WORK_ASSIGNMENT = 31;
const int TAG_NO_WORK_AVAILABLE = 32;
const int TAG_WORK_COMPLETE = 33;

// Estructura para trabajo dinámico
struct WorkItem {
    int image_index;
    std::string image_path;
    size_t image_size;  // Para estimar carga de trabajo
};

// Cola de trabajo thread-safe para el master
class WorkQueue {
private:
    std::queue<WorkItem> pending_work;
    std::mutex queue_mutex;
    
public:
    void addWork(const std::vector<WorkItem>& items);
    bool getNextWork(WorkItem& work);
    bool hasWork() const;
    size_t size() const;
    void clear();
};

// Instancia global de la cola de trabajo
extern WorkQueue g_work_queue;

#ifdef ENABLE_OVERLAP_OPT
// Estructura para procesamiento overlapped
struct OverlapTask {
    int worker_rank;
    std::vector<std::string> image_paths;
    std::vector<unsigned char> lut_data;
    MPI_Request recv_request;
    bool completed;
    
    OverlapTask() : worker_rank(-1), completed(false) {}
};

// Gestor de procesamiento overlapped
class OverlapManager {
private:
    std::vector<OverlapTask> pending_tasks;
    std::vector<MPI_Request> active_requests;
    int max_concurrent_tasks;
    
public:
    OverlapManager(int max_tasks = 4) : max_concurrent_tasks(max_tasks) {}
    
    // Iniciar recepción asíncrona de un worker
    bool startAsyncReceive(int worker_rank, int expected_lut_size);
    
    // Verificar si alguna tarea completó y procesarla
    bool checkAndProcessCompleted();
    
    // Esperar a que todas las tareas pendientes completen
    void waitForAll();
    
    // Limpiar recursos
    void cleanup();
    
    // Verificar si hay espacio para más tareas
    bool hasCapacity() const { return pending_tasks.size() < max_concurrent_tasks; }
};

// Instancia global del overlap manager
extern OverlapManager g_overlap_manager;
#endif
#endif
