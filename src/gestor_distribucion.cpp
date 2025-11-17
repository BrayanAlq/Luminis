#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

using namespace std;

// --- CONSTANTES MPI Y DE LOTE ---
const int TAG_NUEVA_TAREA = 10;
const int TAG_RESULTADO = 20; // Para el envío del array contiguo de LUTs
const int LUT_SIZE = 256;     // Tamaño de cada LUT (unsigned char)

// --- PROTOTIPOS DE FUNCIONES EXTERNAS ---
std::vector<std::string> listarImagenesEnCarpeta(const std::string& carpeta);
std::vector<unsigned char> preprocesarLoteYCalcularLUTs(const std::vector<std::string>& rutas); 
// Usaremos la versión Lote/Asíncrona del Maestro
void aplicarLUTCUDA_Lote(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data); 
void guardarResultadoPlaceholder(); 

// -----------------------------------------------------------------------------
// 1. Ejecuta rol del MAESTRO (Rank 0) - Reparto Estático
// -----------------------------------------------------------------------------
void ejecutarMaestro(int world_size) {
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
}

// -----------------------------------------------------------------------------
// 2. Ejecuta rol del ESCLAVO (Rank > 0) - Procesamiento del Bloque Estático
// -----------------------------------------------------------------------------
void ejecutarEsclavo(int rank) {
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
}