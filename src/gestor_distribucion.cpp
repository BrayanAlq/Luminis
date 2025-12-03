#include "gestor_distribucion.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>
#include "preprocesamiento.hpp"
#include "procesamiento_gpu.hpp"

using namespace std;

// --- CONSTANTES MPI Y DE LOTE ---
const int TAG_NUEVA_TAREA = 10;
const int TAG_RESULTADO = 20; // Para el envío del array contiguo de LUTs
const int LUT_SIZE = 256;     // Tamaño de cada LUT (unsigned char)

// --- PROTOTIPOS DE FUNCIONES EXTERNAS ---
void guardarResultadoPlaceholder();

#ifdef ENABLE_WORK_STEALING_OPT
// Implementación de WorkQueue
WorkQueue g_work_queue;

void WorkQueue::addWork(const std::vector<WorkItem>& items) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (const auto& item : items) {
        pending_work.push(item);
    }
}

bool WorkQueue::getNextWork(WorkItem& work) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (pending_work.empty()) {
        return false;
    }
    work = pending_work.front();
    pending_work.pop();
    return true;
}

bool WorkQueue::hasWork() const {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return !pending_work.empty();
}

size_t WorkQueue::size() const {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return pending_work.size();
}

void WorkQueue::clear() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    while (!pending_work.empty()) {
        pending_work.pop();
    }
}

// Funciones auxiliares para work stealing
std::vector<WorkItem> crearWorkItems(const std::vector<std::string>& rutas);
void ejecutarMaestroDinamico(int world_size);
void ejecutarEsclavoDinamico(int rank);

#ifdef ENABLE_OVERLAP_OPT
// Instancias globales
OverlapManager g_overlap_manager(4); // Máximo 4 tareas concurrentes

// Implementación de OverlapManager
bool OverlapManager::startAsyncReceive(int worker_rank, int expected_lut_size) {
    if (!hasCapacity()) {
        return false;
    }
    
    OverlapTask task;
    task.worker_rank = worker_rank;
    task.lut_data.resize(expected_lut_size);
    task.completed = false;
    
    // Iniciar recepción asíncrona de LUTs
    MPI_Irecv(task.lut_data.data(), expected_lut_size, MPI_UNSIGNED_CHAR, 
              worker_rank, TAG_RESULTADO, MPI_COMM_WORLD, &task.recv_request);
    
    pending_tasks.push_back(task);
    active_requests.push_back(task.recv_request);
    
    cout << "[OVERLAP] Iniciada recepción asíncrona de worker " << worker_rank 
         << " (" << expected_lut_size << " bytes)\n";
    
    return true;
}

bool OverlapManager::checkAndProcessCompleted() {
    if (pending_tasks.empty()) {
        return false;
    }
    
    int completed_index = -1;
    MPI_Status status;
    
    // Verificar si alguna tarea completó
    for (size_t i = 0; i < active_requests.size(); ++i) {
        int flag = 0;
        MPI_Test(&active_requests[i], &flag, &status);
        
        if (flag) {
            completed_index = (int)i;
            break;
        }
    }
    
    if (completed_index == -1) {
        return false; // No hay tareas completadas
    }
    
    // Procesar tarea completada
    OverlapTask& task = pending_tasks[completed_index];
    task.completed = true;
    
    // Recibir información de las imágenes procesadas
    int num_images;
    MPI_Recv(&num_images, 1, MPI_INT, task.worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
    
    std::vector<std::string> processed_paths;
    for (int i = 0; i < num_images; ++i) {
        int path_len;
        MPI_Recv(&path_len, 1, MPI_INT, task.worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
        
        std::string path(path_len, '\0');
        MPI_Recv(&path[0], path_len, MPI_CHAR, task.worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
        processed_paths.push_back(path);
    }
    
    int num_luts = task.lut_data.size() / LUT_SIZE;
    cout << "[OVERLAP] Procesando tarea completada de worker " << task.worker_rank 
         << ": " << num_luts << " LUTs de " << num_images << " imágenes\n";
    
    // Iniciar procesamiento CUDA en paralelo (sin bloquear)
    // NOTA: Esto debería ser asíncrono, pero por ahora mantenemos la llamada síncrona
    // ya que aplicarLUTCUDA_Lote ya usa streams internos
    aplicarLUTCUDA_Lote(processed_paths, task.lut_data);
    
    // Limpiar tarea completada
    pending_tasks.erase(pending_tasks.begin() + completed_index);
    active_requests.erase(active_requests.begin() + completed_index);
    
    return true;
}

void OverlapManager::waitForAll() {
    while (!pending_tasks.empty()) {
        checkAndProcessCompleted();
        
        // Si no hay tareas completadas, esperar un poco
        if (!pending_tasks.empty()) {
            MPI_Status status;
            int completed_index;
            MPI_Waitany(active_requests.size(), active_requests.data(), &completed_index, &status);
            
            if (completed_index != MPI_UNDEFINED) {
                checkAndProcessCompleted();
            }
        }
    }
}

void OverlapManager::cleanup() {
    // Cancelar cualquier recepción pendiente
    for (auto& req : active_requests) {
        MPI_Cancel(&req);
    }
    
    pending_tasks.clear();
    active_requests.clear();
}

// Función modificada para work stealing con overlap
void ejecutarMaestroDinamicoOverlap(int world_size);
#endif
#endif

// -----------------------------------------------------------------------------
// 1. Ejecuta rol del MAESTRO (Rank 0)
// -----------------------------------------------------------------------------
void ejecutarMaestro(int world_size) {
#ifdef ENABLE_WORK_STEALING_OPT
#ifdef ENABLE_OVERLAP_OPT
    cout << "[MASTER] Iniciando gestor de distribución (MODO DINÁMICO - Work Stealing + Overlap).\n";
    ejecutarMaestroDinamicoOverlap(world_size);
#else
    cout << "[MASTER] Iniciando gestor de distribución (MODO DINÁMICO - Work Stealing).\n";
    ejecutarMaestroDinamico(world_size);
#endif
#else
    cout << "[MASTER] Iniciando gestor de distribución (MODO ESTATICO).\n";

    vector<string> rutas_globales = listarImagenesEnCarpeta("data/input/");
    int total = (int)rutas_globales.size();
    int numWorkers = world_size - 1;
    
    if (numWorkers == 0) { cout << "[MASTER] No hay esclavos. Fin.\n"; return; }
    if (total == 0) { cout << "[MASTER] No hay imágenes para procesar. Fin.\n"; return; }

    int base = total / numWorkers;
    int rem = total % numWorkers;
    MPI_Status status;
    
    // Almacenamiento temporal para el Maestro (rutas esperadas de cada Worker)
    std::vector<std::vector<std::string>> rutas_esperadas(world_size); 
    
    // 1. Reparto Estático: Enviar a cada worker su rango [inicio, fin)
    int offset = 0;
    for (int rank = 1; rank < world_size; ++rank) {
        int fin = offset + base + ((rank - 1) < rem ? 1 : 0);
        
        if (offset < total) { 
            int datos[2] = { offset, fin };

            // ENVIAR RANGO [inicio, fin)
            MPI_Send(datos, 2, MPI_INT, rank, TAG_NUEVA_TAREA, MPI_COMM_WORLD);
            cout << "[MASTER] Enviado rango estático ["<< offset <<","<< fin <<") al worker " << rank << "\n";
            
            // Almacenar las rutas que este Worker debe procesar
            for (int i = offset; i < fin; ++i) {
                rutas_esperadas[rank].push_back(rutas_globales[i]);
            }
            offset = fin;
        } else {
             // Enviar señal de terminación a Workers sobrantes
             int datos[2] = { -1, -1 };
             MPI_Send(datos, 2, MPI_INT, rank, TAG_NUEVA_TAREA, MPI_COMM_WORLD);
        }
    }

    // 2. Recepción Única: Recibir LUTs de todos los Workers
    for (int rank = 1; rank < world_size; ++rank) {
        // Ignoramos Workers que no recibieron trabajo
        if (rutas_esperadas[rank].empty()) continue; 
        
        int num_bytes_lut;
        
        // A. Recibir LUTs (usamos MPI_Probe para conocer el tamaño exacto del lote grande)
        MPI_Probe(rank, TAG_RESULTADO, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &num_bytes_lut);
        std::vector<unsigned char> lut_data(num_bytes_lut);
        
        MPI_Recv(lut_data.data(), num_bytes_lut, MPI_UNSIGNED_CHAR, rank, TAG_RESULTADO, MPI_COMM_WORLD, &status);
        
        int num_luts_recibidas = num_bytes_lut / LUT_SIZE;
        cout << "[MASTER] Recibido lote de " << num_luts_recibidas << " LUTs del Worker " << rank << ".\n";
        
        // B. Aplicación de la LUTs con CUDA (Llamada al código asíncrono)
        aplicarLUTCUDA_Lote(rutas_esperadas[rank], lut_data); 
    }
    
guardarResultadoPlaceholder();
    cout << "[MASTER] Trabajo finalizado.\n";
#endif
}

// -----------------------------------------------------------------------------
// 2. Ejecuta rol del ESCLAVO (Rank > 0)
// -----------------------------------------------------------------------------
void ejecutarEsclavo(int rank) {
#ifdef ENABLE_WORK_STEALING_OPT
    cout << "[WORKER " << rank << "] Iniciando modo dinámico (Work Stealing).\n";
    ejecutarEsclavoDinamico(rank);
#else
    cout << "[WORKER " << rank << "] Esperando rango estático del maestro...\n";
    
    int datos[2];
    MPI_Status status;
    
    // 1. Recibir Tarea (Rango de índices [inicio, fin))
    MPI_Recv(datos, 2, MPI_INT, 0, TAG_NUEVA_TAREA, MPI_COMM_WORLD, &status);
    int inicio = datos[0], fin = datos[1];
    
    // Si recibe un rango inválido, termina.
    if (inicio < 0 || fin < 0 || inicio >= fin) {
        cout << "[WORKER " << rank << "] No hay trabajo asignado. Finalizando.\n";
        return;
    }
    
    cout << "[WORKER " << rank << "] Recibido rango estático ["<< inicio <<","<< fin <<")\n";

    // 2. Obtener la lista completa y tomar su sub-rango
    vector<string> rutas_globales = listarImagenesEnCarpeta("data/input/");
    
    // Verificar límites 
    fin = std::min(fin, (int)rutas_globales.size()); 
    
    std::vector<std::string> rutas_a_procesar(
        rutas_globales.begin() + inicio, 
        rutas_globales.begin() + fin
    );

    // 3. Cálculo de las LUTs para el lote grande (Usa OpenMP internamente)
    std::vector<unsigned char> lut_data = preprocesarLoteYCalcularLUTs(rutas_a_procesar); 

    // 4. Envío Único de la LUTs al Maestro
    
    MPI_Send(lut_data.data(), lut_data.size(), MPI_UNSIGNED_CHAR, 0, TAG_RESULTADO, MPI_COMM_WORLD);
    
cout << "[WORKER " << rank << "] Lote grande de LUTs enviado al Maestro (" << lut_data.size() << " bytes).\n";
#endif
}

#ifdef ENABLE_WORK_STEALING_OPT
// -----------------------------------------------------------------------------
// IMPLEMENTACIÓN DEL WORK STEALING DINÁMICO
// -----------------------------------------------------------------------------

// Función para crear WorkItems con información de tamaño
std::vector<WorkItem> crearWorkItems(const std::vector<std::string>& rutas) {
    std::vector<WorkItem> items;
    
    for (size_t i = 0; i < rutas.size(); ++i) {
        WorkItem item;
        item.image_index = (int)i;
        item.image_path = rutas[i];
        
        // Estimar tamaño de la imagen (carga rápida sin cargar completamente)
        cv::Mat temp_img = cv::imread(rutas[i], cv::IMREAD_GRAYSCALE);
        if (!temp_img.empty()) {
            item.image_size = temp_img.rows * temp_img.cols;
        } else {
            item.image_size = 1024 * 1024; // Tamaño por defecto si no se puede leer
        }
        
        items.push_back(item);
    }
    
    return items;
}

// -----------------------------------------------------------------------------
// Master Dinámico con Work Stealing
// -----------------------------------------------------------------------------
void ejecutarMaestroDinamico(int world_size) {
    vector<string> rutas_globales = listarImagenesEnCarpeta("data/input/");
    int total = (int)rutas_globales.size();
    int numWorkers = world_size - 1;
    
    if (numWorkers == 0) { cout << "[MASTER] No hay workers. Fin.\n"; return; }
    if (total == 0) { cout << "[MASTER] No hay imágenes. Fin.\n"; return; }
    
    // 1. Crear y poblar la cola de trabajo
    std::vector<WorkItem> work_items = crearWorkItems(rutas_globales);
    g_work_queue.addWork(work_items);
    
    cout << "[MASTER] Cola de trabajo inicializada con " << g_work_queue.size() << " imágenes.\n";
    
    // 2. Enviar trabajo inicial a todos los workers
    for (int rank = 1; rank < world_size; ++rank) {
        WorkItem work;
        if (g_work_queue.getNextWork(work)) {
            // Enviar trabajo al worker
            int work_data[3] = {work.image_index, (int)work.image_size, 0}; // 0 = más trabajo disponible
            MPI_Send(work_data, 3, MPI_INT, rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
            
            // Enviar ruta de la imagen
            int path_len = (int)work.image_path.length();
            MPI_Send(&path_len, 1, MPI_INT, rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
            MPI_Send(work.image_path.c_str(), path_len, MPI_CHAR, rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
            
            cout << "[MASTER] Enviada imagen " << work.image_index << " (" << work.image_size << " px) al worker " << rank << "\n";
        }
    }
    
    // 3. Procesar solicitudes de trabajo dinámicamente
    int completed_workers = 0;
    std::vector<std::vector<std::string>> worker_results(world_size);
    
    while (completed_workers < numWorkers) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if (status.MPI_TAG == TAG_REQUEST_WORK) {
            // Worker solicita más trabajo
            int worker_rank;
            MPI_Recv(&worker_rank, 1, MPI_INT, status.MPI_SOURCE, TAG_REQUEST_WORK, MPI_COMM_WORLD, &status);
            
            WorkItem work;
            if (g_work_queue.getNextWork(work)) {
                // Hay más trabajo disponible
                int work_data[3] = {work.image_index, (int)work.image_size, 0};
                MPI_Send(work_data, 3, MPI_INT, worker_rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
                
                int path_len = (int)work.image_path.length();
                MPI_Send(&path_len, 1, MPI_INT, worker_rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
                MPI_Send(work.image_path.c_str(), path_len, MPI_CHAR, worker_rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
                
                cout << "[MASTER] Enviada imagen " << work.image_index << " al worker " << worker_rank << " (work stealing)\n";
            } else {
                // No hay más trabajo
                int no_work[3] = {-1, -1, -1};
                MPI_Send(no_work, 3, MPI_INT, worker_rank, TAG_NO_WORK_AVAILABLE, MPI_COMM_WORLD);
                cout << "[MASTER] No hay más trabajo para worker " << worker_rank << "\n";
            }
        }
        else if (status.MPI_TAG == TAG_RESULTADO) {
            // Worker envía resultados
            int worker_rank = status.MPI_SOURCE;
            
            // Recibir LUTs
            int num_bytes_lut;
            MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &num_bytes_lut);
            std::vector<unsigned char> lut_data(num_bytes_lut);
            MPI_Recv(lut_data.data(), num_bytes_lut, MPI_UNSIGNED_CHAR, worker_rank, TAG_RESULTADO, MPI_COMM_WORLD, &status);
            
            // Recibir información de las imágenes procesadas
            int num_images;
            MPI_Recv(&num_images, 1, MPI_INT, worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
            
            std::vector<std::string> processed_paths;
            for (int i = 0; i < num_images; ++i) {
                int path_len;
                MPI_Recv(&path_len, 1, MPI_INT, worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
                
                std::string path(path_len, '\0');
                MPI_Recv(&path[0], path_len, MPI_CHAR, worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
                processed_paths.push_back(path);
            }
            
            int num_luts = num_bytes_lut / LUT_SIZE;
            cout << "[MASTER] Recibidas " << num_luts << " LUTs de " << num_images << " imágenes del worker " << worker_rank << "\n";
            
            // Aplicar LUTs con CUDA
            aplicarLUTCUDA_Lote(processed_paths, lut_data);
            
            // Worker solicita más trabajo
            MPI_Send(&worker_rank, 1, MPI_INT, 0, TAG_REQUEST_WORK, MPI_COMM_WORLD);
        }
    }
    
    cout << "[MASTER] Todos los workers completaron. Trabajo finalizado.\n";
    g_work_queue.clear();
}

// -----------------------------------------------------------------------------
// Worker Dinámico con Work Stealing
// -----------------------------------------------------------------------------
void ejecutarEsclavoDinamico(int rank) {
    std::vector<std::string> processed_images;
    std::vector<unsigned char> accumulated_luts;
    
    while (true) {
        // Solicitar trabajo al master
        MPI_Send(&rank, 1, MPI_INT, 0, TAG_REQUEST_WORK, MPI_COMM_WORLD);
        
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if (status.MPI_TAG == TAG_NO_WORK_AVAILABLE) {
            // No hay más trabajo, enviar resultados acumulados y terminar
            if (!accumulated_luts.empty()) {
                MPI_Send(accumulated_luts.data(), (int)accumulated_luts.size(), MPI_UNSIGNED_CHAR, 0, TAG_RESULTADO, MPI_COMM_WORLD);
                
                int num_images = (int)processed_images.size();
                MPI_Send(&num_images, 1, MPI_INT, 0, TAG_WORK_COMPLETE, MPI_COMM_WORLD);
                
                for (const auto& path : processed_images) {
                    int path_len = (int)path.length();
                    MPI_Send(&path_len, 1, MPI_INT, 0, TAG_WORK_COMPLETE, MPI_COMM_WORLD);
                    MPI_Send(path.c_str(), path_len, MPI_CHAR, 0, TAG_WORK_COMPLETE, MPI_COMM_WORLD);
                }
                
                cout << "[WORKER " << rank << "] Enviados resultados finales: " << num_images << " imágenes\n";
            }
            break;
        }
        else if (status.MPI_TAG == TAG_WORK_ASSIGNMENT) {
            // Recibir asignación de trabajo
            int work_data[3];
            MPI_Recv(work_data, 3, MPI_INT, 0, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            
            if (work_data[0] == -1) {
                // Señal de terminación
                break;
            }
            
            // Recibir ruta de la imagen
            int path_len;
            MPI_Recv(&path_len, 1, MPI_INT, 0, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            
            std::string path(path_len, '\0');
            MPI_Recv(&path[0], path_len, MPI_CHAR, 0, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD, &status);
            
            cout << "[WORKER " << rank << "] Procesando imagen: " << path << "\n";
            
            // Procesar imagen y calcular LUT
            std::vector<unsigned char> lut = preprocesarImagenYCalcularLUT(path);
            
            if (!lut.empty()) {
                processed_images.push_back(path);
                accumulated_luts.insert(accumulated_luts.end(), lut.begin(), lut.end());
            }
            
            // Enviar resultados si tenemos un lote significativo o si no hay más trabajo
            if (accumulated_luts.size() >= LUT_SIZE * 10 || work_data[2] == 1) {
                MPI_Send(accumulated_luts.data(), (int)accumulated_luts.size(), MPI_UNSIGNED_CHAR, 0, TAG_RESULTADO, MPI_COMM_WORLD);
                
                int num_images = (int)processed_images.size();
                MPI_Send(&num_images, 1, MPI_INT, 0, TAG_WORK_COMPLETE, MPI_COMM_WORLD);
                
                for (const auto& proc_path : processed_images) {
                    int proc_path_len = (int)proc_path.length();
                    MPI_Send(&proc_path_len, 1, MPI_INT, 0, TAG_WORK_COMPLETE, MPI_COMM_WORLD);
                    MPI_Send(proc_path.c_str(), proc_path_len, MPI_CHAR, 0, TAG_WORK_COMPLETE, MPI_COMM_WORLD);
                }
                
                cout << "[WORKER " << rank << "] Enviado lote: " << num_images << " imágenes\n";
                
                // Limpiar para siguiente lote
                processed_images.clear();
                accumulated_luts.clear();
            }
        }
    }
    
    cout << "[WORKER " << rank << "] Trabajo completado.\n";
}

#ifdef ENABLE_OVERLAP_OPT
// -----------------------------------------------------------------------------
// Master Dinámico con Work Stealing + Overlap Comunicación-Cómputo
// -----------------------------------------------------------------------------
void ejecutarMaestroDinamicoOverlap(int world_size) {
    vector<string> rutas_globales = listarImagenesEnCarpeta("data/input/");
    int total = (int)rutas_globales.size();
    int numWorkers = world_size - 1;
    
    if (numWorkers == 0) { cout << "[MASTER] No hay workers. Fin.\n"; return; }
    if (total == 0) { cout << "[MASTER] No hay imágenes. Fin.\n"; return; }
    
    // 1. Crear y poblar la cola de trabajo
    std::vector<WorkItem> work_items = crearWorkItems(rutas_globales);
    g_work_queue.addWork(work_items);
    
    cout << "[MASTER] Cola de trabajo inicializada con " << g_work_queue.size() << " imágenes.\n";
    cout << "[MASTER] Iniciando modo overlapped con capacidad para " << g_overlap_manager.hasCapacity() << " tareas concurrentes.\n";
    
    // 2. Enviar trabajo inicial a todos los workers
    for (int rank = 1; rank < world_size; ++rank) {
        WorkItem work;
        if (g_work_queue.getNextWork(work)) {
            // Enviar trabajo al worker
            int work_data[3] = {work.image_index, (int)work.image_size, 0}; // 0 = más trabajo disponible
            MPI_Send(work_data, 3, MPI_INT, rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
            
            // Enviar ruta de la imagen
            int path_len = (int)work.image_path.length();
            MPI_Send(&path_len, 1, MPI_INT, rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
            MPI_Send(work.image_path.c_str(), path_len, MPI_CHAR, rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
            
            cout << "[MASTER] Enviada imagen " << work.image_index << " (" << work.image_size << " px) al worker " << rank << "\n";
        }
    }
    
    // 3. Bucle principal con overlap comunicación-cómputo
    int completed_workers = 0;
    bool workers_active[numWorkers + 1];
    std::fill(workers_active, workers_active + numWorkers + 1, true);
    
    while (completed_workers < numWorkers) {
        // 3.1. Procesar tareas completadas (overlap)
        g_overlap_manager.checkAndProcessCompleted();
        
        // 3.2. Verificar solicitudes de trabajo y resultados
        MPI_Status status;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        
        if (flag) {
            if (status.MPI_TAG == TAG_REQUEST_WORK) {
                // Worker solicita más trabajo
                int worker_rank;
                MPI_Recv(&worker_rank, 1, MPI_INT, status.MPI_SOURCE, TAG_REQUEST_WORK, MPI_COMM_WORLD, &status);
                
                WorkItem work;
                if (g_work_queue.getNextWork(work)) {
                    // Hay más trabajo disponible
                    int work_data[3] = {work.image_index, (int)work.image_size, 0};
                    MPI_Send(work_data, 3, MPI_INT, worker_rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
                    
                    int path_len = (int)work.image_path.length();
                    MPI_Send(&path_len, 1, MPI_INT, worker_rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
                    MPI_Send(work.image_path.c_str(), path_len, MPI_CHAR, worker_rank, TAG_WORK_ASSIGNMENT, MPI_COMM_WORLD);
                    
                    cout << "[MASTER] Enviada imagen " << work.image_index << " al worker " << worker_rank << " (work stealing)\n";
                } else {
                    // No hay más trabajo
                    int no_work[3] = {-1, -1, -1};
                    MPI_Send(no_work, 3, MPI_INT, worker_rank, TAG_NO_WORK_AVAILABLE, MPI_COMM_WORLD);
                    workers_active[worker_rank] = false;
                    cout << "[MASTER] No hay más trabajo para worker " << worker_rank << "\n";
                }
            }
            else if (status.MPI_TAG == TAG_RESULTADO) {
                // Worker envía resultados - iniciar recepción asíncrona si hay capacidad
                int worker_rank = status.MPI_SOURCE;
                
                // Obtener tamaño del mensaje
                int num_bytes_lut;
                MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &num_bytes_lut);
                
                // Intentar iniciar recepción asíncrona
                if (g_overlap_manager.startAsyncReceive(worker_rank, num_bytes_lut)) {
                    cout << "[MASTER] Iniciada recepción asíncrona de worker " << worker_rank << "\n";
                } else {
                    // No hay capacidad, procesar síncronamente
                    std::vector<unsigned char> lut_data(num_bytes_lut);
                    MPI_Recv(lut_data.data(), num_bytes_lut, MPI_UNSIGNED_CHAR, worker_rank, TAG_RESULTADO, MPI_COMM_WORLD, &status);
                    
                    // Recibir información de las imágenes procesadas
                    int num_images;
                    MPI_Recv(&num_images, 1, MPI_INT, worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
                    
                    std::vector<std::string> processed_paths;
                    for (int i = 0; i < num_images; ++i) {
                        int path_len;
                        MPI_Recv(&path_len, 1, MPI_INT, worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
                        
                        std::string path(path_len, '\0');
                        MPI_Recv(&path[0], path_len, MPI_CHAR, worker_rank, TAG_WORK_COMPLETE, MPI_COMM_WORLD, &status);
                        processed_paths.push_back(path);
                    }
                    
                    int num_luts = num_bytes_lut / LUT_SIZE;
                    cout << "[MASTER] Procesando síncronamente " << num_luts << " LUTs de " << num_images << " imágenes del worker " << worker_rank << "\n";
                    
                    // Aplicar LUTs con CUDA
                    aplicarLUTCUDA_Lote(processed_paths, lut_data);
                    
                    // Worker solicita más trabajo
                    if (workers_active[worker_rank]) {
                        MPI_Send(&worker_rank, 1, MPI_INT, 0, TAG_REQUEST_WORK, MPI_COMM_WORLD);
                    }
                }
            }
        }
        
        // 3.3. Pequeña pausa para evitar busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // 4. Esperar a que todas las tareas asíncronas completen
    cout << "[MASTER] Esperando finalización de tareas pendientes...\n";
    g_overlap_manager.waitForAll();
    
    cout << "[MASTER] Todos los workers completaron. Trabajo finalizado con overlap.\n";
    g_work_queue.clear();
    g_overlap_manager.cleanup();
}
#endif
#endif