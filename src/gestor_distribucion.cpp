// gestor_distribucion.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

// prototipos de funciones de otros módulos (se definen en sus .cpp/.cu)
std::vector<std::string> listarImagenesEnCarpeta(const std::string& carpeta);

std::vector<int> preprocesarImagenYCalcularHist(const std::string& ruta);
std::vector<int> preprocesarLote(const std::vector<std::string>& rutas, int inicio, int fin);

void enviarHistogramasAlMaestro(const std::vector<int>& hist, int destino, int tag);
std::vector<std::vector<int>> recibirHistogramasDesdeEsclavos(int world_size);

void postprocesar_en_GPU(const std::vector<int>& histGlobal);
void guardarResultadoPlaceholder(); // placeholder

// Ejecuta rol del maestro
void ejecutarMaestro(int world_size) {
    cout << "[MASTER] Iniciando gestor de distribución.\n";

    // Lista local de imágenes (puede leerse y repartirse)
    vector<string> rutas = listarImagenesEnCarpeta("data/input/");
    int total = (int)rutas.size();
    if (world_size <= 1) {
        cout << "[MASTER] No hay esclavos. Fin.\n";
        return;
    }

    // Reparto simple: rango por nodo (maestro no procesa imágenes)
    int numWorkers = world_size - 1;
    int base = total / numWorkers;
    int rem = total % numWorkers;

    // enviar a cada worker su rango (inicio, fin) usando tags
    for (int rank = 1; rank < world_size; ++rank) {
        int inicio = (rank - 1) * base + min(rank - 1, rem);
        int fin = inicio + base + ((rank - 1) < rem ? 1 : 0);
        int datos[2] = { inicio, fin };
        MPI_Send(datos, 2, MPI_INT, rank, 10, MPI_COMM_WORLD);
        cout << "[MASTER] Enviado rango ["<< inicio <<","<< fin <<") al worker " << rank << "\n";
    }

    // recibir histogramas de los workers
    auto todosHistogramas = recibirHistogramasDesdeEsclavos(world_size);
    // combinar en histograma global
    vector<int> histGlobal(256, 0);
    for (auto &h : todosHistogramas) {
        for (int i = 0; i < 256; ++i) histGlobal[i] += h[i];
    }

    cout << "[MASTER] Histograma global preparado. Enviando a GPU...\n";
    postprocesar_en_GPU(histGlobal);

    // guardar resultado (placeholder)
    guardarResultadoPlaceholder();

    cout << "[MASTER] Trabajo finalizado.\n";
}

// Ejecuta rol del esclavo
void ejecutarEsclavo(int rank) {
    cout << "[WORKER " << rank << "] Esperando rango del maestro...\n";
    int datos[2];
    MPI_Status status;
    MPI_Recv(datos, 2, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    int inicio = datos[0], fin = datos[1];
    cout << "[WORKER " << rank << "] Recibido rango ["<< inicio <<","<< fin <<")\n";

    // obtener lista completa (cada worker puede leer la lista y tomar su parte)
    vector<string> rutas = listarImagenesEnCarpeta("data/input/");
    if (inicio < 0) inicio = 0;
    if (fin > (int)rutas.size()) fin = rutas.size();

    // preprocesar las imágenes asignadas y obtener histograma global local
    vector<int> histGlobal = preprocesarLote(rutas, inicio, fin);

    // enviar histograma al maestro
    enviarHistogramasAlMaestro(histGlobal, 0, rank);
    cout << "[WORKER " << rank << "] Histograma enviado al maestro.\n";
}
