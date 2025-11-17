#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>

using namespace std;

// --- CONSTANTES MPI Y DE LOTE ---
const int BATCH_SIZE = 5; // <--- Tamaño del lote (ajustable para optimizar)
const int TAG_NUEVA_TAREA = 10;
const int TAG_RESULTADO = 20;
const int TAG_RUTA_COMPLETADA = 21; // Usaremos TAG_RESULTADO + 1
const int LUT_SIZE = 256;

// --- PROTOTIPOS DE FUNCIONES EXTERNAS ---
std::vector<std::string> listarImagenesEnCarpeta(const std::string& carpeta);
std::vector<unsigned char> preprocesarLoteYCalcularLUTs(const std::vector<std::string>& rutas); 
void aplicarLUTCUDA_Lote(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data); 
void guardarResultadoPlaceholder(); 

// --- FUNCIONES INTERNAS DE UTILIDAD ---

// Ayudante: Convierte un vector de rutas a una cadena separada por '\n'
std::string rutasAVector(const std::vector<std::string>& rutas) {
    if (rutas.empty()) return "";
    std::stringstream ss;
    for (const auto& r : rutas) ss << r << "\n";
    return ss.str();
}

// Ayudante: Convierte la cadena de rutas recibida de vuelta a un vector
std::vector<std::string> vectorARutas(const std::string& str) {
    std::vector<std::string> rutas;
    if (str.empty()) return rutas;
    std::stringstream ss(str);
    std::string ruta;
    while (std::getline(ss, ruta, '\n')) {
        if (!ruta.empty()) rutas.push_back(ruta);
    }
    return rutas;
}

// -----------------------------------------------------------------------------
// 1. Ejecuta rol del MAESTRO (Rank 0)
// -----------------------------------------------------------------------------
void ejecutarMaestro(int world_size) {
    cout << "[MASTER] Iniciando gestor de distribución (MODO LOTE, size=" << BATCH_SIZE << ").\n";

    vector<string> rutas_globales = listarImagenesEnCarpeta("data/input/");
    int numWorkers = world_size - 1;
    if (numWorkers == 0) { cout << "[MASTER] No hay esclavos. Fin.\n"; return; }

    int numTareas = (int)rutas_globales.size();
    int tareasEnviadas = 0;
    int tareasCompletadas = 0;
    MPI_Status status;
    int longitud_buffer_max = (BATCH_SIZE * (256 + 1));

    // 2. Reparto inicial: Una tarea (lote de rutas) a cada Worker
    for (int rank = 1; rank < world_size; ++rank) {
        if (tareasEnviadas < numTareas) {
            int fin = std::min(tareasEnviadas + BATCH_SIZE, numTareas);
            std::vector<std::string> lote(rutas_globales.begin() + tareasEnviadas, rutas_globales.begin() + fin);
            std::string rutas_str = rutasAVector(lote);
            
            MPI_Send(rutas_str.c_str(), rutas_str.length() + 1, MPI_CHAR, rank, TAG_NUEVA_TAREA, MPI_COMM_WORLD);
            cout << "[MASTER] Enviado lote [" << tareasEnviadas << "-" << fin << ") al Worker " << rank << "\n";
            tareasEnviadas = fin;
        }
    }
    
    // 3. Pool of Workers: Bucle de Recepción y Reenvío
    while (tareasCompletadas < numTareas) {
        char buffer_rutas_completadas[longitud_buffer_max];
        int num_bytes_lut;
        
        // A. Esperar y Recibir LUTs
        MPI_Probe(MPI_ANY_SOURCE, TAG_RESULTADO, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &num_bytes_lut);
        std::vector<unsigned char> lut_data(num_bytes_lut);
        int worker_rank = status.MPI_SOURCE;
        
        MPI_Recv(lut_data.data(), num_bytes_lut, MPI_UNSIGNED_CHAR, worker_rank, TAG_RESULTADO, MPI_COMM_WORLD, &status);
        
        // B. Recibir las rutas completadas
        MPI_Recv(buffer_rutas_completadas, longitud_buffer_max, MPI_CHAR, worker_rank, TAG_RUTA_COMPLETADA, MPI_COMM_WORLD, &status);
        std::vector<std::string> rutas_completadas = vectorARutas(buffer_rutas_completadas);

        int lote_completado = (int)rutas_completadas.size();
        tareasCompletadas += lote_completado;
        cout << "[MASTER] Recibido lote de " << lote_completado << " items de Worker " << worker_rank 
             << ". Completadas: " << tareasCompletadas << "/" << numTareas << "\n";
        
        // C. Aplicación de la LUTs con CUDA
        aplicarLUTCUDA_Lote(rutas_completadas, lut_data); 

        // D. Asignar nueva tarea al Worker que acaba de terminar
        if (tareasEnviadas < numTareas) {
            int fin = std::min(tareasEnviadas + BATCH_SIZE, numTareas);
            std::vector<std::string> lote(rutas_globales.begin() + tareasEnviadas, rutas_globales.begin() + fin);
            std::string rutas_str = rutasAVector(lote);
            
            MPI_Send(rutas_str.c_str(), rutas_str.length() + 1, MPI_CHAR, worker_rank, TAG_NUEVA_TAREA, MPI_COMM_WORLD);
            cout << "[MASTER] Enviado NUEVO lote [" << tareasEnviadas << "-" << fin << ") al Worker " << worker_rank << "\n";
            tareasEnviadas = fin;
        } else {
            // Enviar señal de terminación (cadena vacía)
            MPI_Send("", 1, MPI_CHAR, worker_rank, TAG_NUEVA_TAREA, MPI_COMM_WORLD);
        }
    }
    
    guardarResultadoPlaceholder();
    cout << "[MASTER] Trabajo finalizado.\n";
}

// -----------------------------------------------------------------------------
// 2. Ejecuta rol del ESCLAVO (Rank > 0)
// -----------------------------------------------------------------------------
void ejecutarEsclavo(int rank) {
    cout << "[WORKER " << rank << "] Iniciando Worker (MODO LOTE).\n";
    MPI_Status status;
    bool continuar = true;
    int longitud_buffer_max = (BATCH_SIZE * (256 + 1));

    while (continuar) {
        char buffer_rutas[longitud_buffer_max]; 
        int longitud_buffer;

        // 1. Recibir Tarea (Lote de Rutas)
        MPI_Recv(buffer_rutas, longitud_buffer_max, MPI_CHAR, 0, TAG_NUEVA_TAREA, MPI_COMM_WORLD, &status);
        
        MPI_Get_count(&status, MPI_CHAR, &longitud_buffer);

        if (longitud_buffer <= 1) { 
            continuar = false;
            break;
        }

        std::string rutas_str(buffer_rutas);
        std::vector<std::string> rutas_a_procesar = vectorARutas(rutas_str);
        
        cout << "[WORKER " << rank << "] Procesando lote de " << rutas_a_procesar.size() << " imágenes.\n";

        // 2. Cálculo de las LUTs (Lógica en preprocesamiento.cpp)
        std::vector<unsigned char> lut_data = preprocesarLoteYCalcularLUTs(rutas_a_procesar); 

        // 3. Envío de la LUTs y Rutas al Maestro
        
        // A. Enviar el array contiguo de LUTs
        MPI_Send(lut_data.data(), lut_data.size(), MPI_UNSIGNED_CHAR, 0, TAG_RESULTADO, MPI_COMM_WORLD);
        
        // B. Enviar las rutas de los archivos completados (mantenemos la cadena original para simplificar)
        MPI_Send(rutas_str.c_str(), rutas_str.length() + 1, MPI_CHAR, 0, TAG_RUTA_COMPLETADA, MPI_COMM_WORLD);
        
        cout << "[WORKER " << rank << "] Lote de LUTs y rutas enviado. Esperando nueva tarea...\n";
    }
    cout << "[WORKER " << rank << "] Finalizando.\n";
}