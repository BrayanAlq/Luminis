#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // Para std::min

using namespace std;

// --- CONSTANTES MPI ---
// Usaremos estas etiquetas para que la comunicación sea clara
const int TAG_NUEVA_TAREA = 10;      // Maestro a Worker: Envía la ruta de la imagen a procesar
const int TAG_LUT = 20;              // Worker a Maestro: Envía el array de la LUT
const int TAG_RUTA_COMPLETADA = 21;  // Worker a Maestro: Envía la ruta del archivo procesado
const int TAG_FIN = 30;              // Maestro a Worker: Señal de terminación
const int LUT_SIZE = 256;            // El tamaño de la LUT siempre es 256 (unsigned char)

// --- PROTOTIPOS DE FUNCIONES EXTERNAS ---
// Estas funciones DEBES definirlas en otros archivos (.cpp, .cu o .hpp)
std::vector<std::string> listarImagenesEnCarpeta(const std::string& carpeta);
std::vector<unsigned char> preprocesarImagenYCalcularLUT(const std::string& ruta); // Lógica del Worker (CPU)
void aplicarLUTCUDA(const std::string& ruta, const std::vector<unsigned char>& lut); // Lógica del Maestro (GPU)
void guardarResultadoPlaceholder(); // Placeholder

// --- FUNCIONES INTERNAS ---

// 1. Ejecuta rol del MAESTRO (Rank 0)
void ejecutarMaestro(int world_size) {
    cout << "[MASTER] Iniciando gestor de distribución (MODO LUT: Pool of Workers).\n";

    vector<string> rutas = listarImagenesEnCarpeta("data/input/");
    int numWorkers = world_size - 1;
    if (numWorkers == 0) {
        cout << "[MASTER] No hay esclavos. Fin.\n";
        return;
    }

    int numTareas = (int)rutas.size();
    int tareasEnviadas = 0;
    int tareasCompletadas = 0;
    MPI_Status status;
    
    // 2. Reparto inicial: Una tarea a cada Worker disponible
    for (int rank = 1; rank < world_size; ++rank) {
        if (tareasEnviadas < numTareas) {
            const string& ruta = rutas[tareasEnviadas];
            // ENVIAR TAREA (Ruta)
            MPI_Send(ruta.c_str(), ruta.length() + 1, MPI_CHAR, rank, TAG_NUEVA_TAREA, MPI_COMM_WORLD);
            cout << "[MASTER] Enviada tarea '" << ruta << "' al Worker " << rank << "\n";
            tareasEnviadas++;
        }
    }
    
    // 3. Pool of Workers: Bucle de Recepción y Reenvío
    while (tareasCompletadas < numTareas) {
        char ruta_buffer[256]; 
        vector<unsigned char> lut_recv(LUT_SIZE);

        // A. Esperar y Recibir la LUT (Bloqueante: asegura que un worker está listo)
        MPI_Recv(lut_recv.data(), LUT_SIZE, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, TAG_LUT, MPI_COMM_WORLD, &status);
        int worker_rank = status.MPI_SOURCE;

        // B. Recibir el nombre de la imagen que corresponde a esta LUT
        MPI_Recv(ruta_buffer, 256, MPI_CHAR, worker_rank, TAG_RUTA_COMPLETADA, MPI_COMM_WORLD, &status);
        string ruta_completada(ruta_buffer);

        tareasCompletadas++;
        cout << "[MASTER] Recibida LUT para '" << ruta_completada << "' del Worker " << worker_rank 
             << ". Completadas: " << tareasCompletadas << "/" << numTareas << "\n";
        
        // C. Aplicación de la LUT con CUDA (Usa la GPU del Maestro)
        aplicarLUTCUDA(ruta_completada, lut_recv); 

        // D. Asignar nueva tarea al Worker que acaba de terminar (si hay más)
        if (tareasEnviadas < numTareas) {
            const string& ruta_nueva = rutas[tareasEnviadas];
            MPI_Send(ruta_nueva.c_str(), ruta_nueva.length() + 1, MPI_CHAR, worker_rank, TAG_NUEVA_TAREA, MPI_COMM_WORLD);
            cout << "[MASTER] Enviada NUEVA tarea '" << ruta_nueva << "' al Worker " << worker_rank << "\n";
            tareasEnviadas++;
        }
    }

    // 4. Señal de terminación: Enviar la señal de fin a todos los Workers
    // Ya que los Workers que terminaron sus tareas se quedan esperando la próxima tarea.
    for (int rank = 1; rank < world_size; ++rank) {
        // Enviar cadena vacía (longitud 1) como señal de fin
        MPI_Send("", 1, MPI_CHAR, rank, TAG_NUEVA_TAREA, MPI_COMM_WORLD);
    }
    
    guardarResultadoPlaceholder();
    cout << "[MASTER] Trabajo finalizado.\n";
}

// 2. Ejecuta rol del ESCLAVO (Rank > 0)
void ejecutarEsclavo(int rank) {
    cout << "[WORKER " << rank << "] Iniciando Worker (MODO LUT).\n";
    MPI_Status status;
    bool continuar = true;

    while (continuar) {
        char ruta_buffer[256]; 
        int longitud_ruta;

        // 1. Recibir Tarea (Ruta de la imagen)
        MPI_Recv(ruta_buffer, 256, MPI_CHAR, 0, TAG_NUEVA_TAREA, MPI_COMM_WORLD, &status);
        
        MPI_Get_count(&status, MPI_CHAR, &longitud_ruta);

        // Si la longitud es 1 (solo el terminador \0), es la señal de fin
        if (longitud_ruta <= 1) { 
            cout << "[WORKER " << rank << "] Recibida señal de finalización. Terminando.\n";
            continuar = false;
            break;
        }

        std::string ruta_imagen(ruta_buffer);
        cout << "[WORKER " << rank << "] Procesando: " << ruta_imagen << "\n";

        // 2. Cálculo de la LUT (Lógica en preprocesamiento.cpp)
        // Utiliza OpenMP internamente si está implementado.
        std::vector<unsigned char> lut = preprocesarImagenYCalcularLUT(ruta_imagen); 

        // 3. Enviar la LUT (datos) al Maestro
        MPI_Send(lut.data(), LUT_SIZE, MPI_UNSIGNED_CHAR, 0, TAG_LUT, MPI_COMM_WORLD);
        
        // 4. Enviar el nombre de la imagen que corresponde a la LUT
        MPI_Send(ruta_imagen.c_str(), ruta_imagen.length() + 1, MPI_CHAR, 0, TAG_RUTA_COMPLETADA, MPI_COMM_WORLD);
        
        cout << "[WORKER " << rank << "] LUT y ruta enviadas al Maestro. Esperando nueva tarea...\n";
    }
}