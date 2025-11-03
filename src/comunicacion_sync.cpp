// comunicacion.cpp
#include <mpi.h>
#include <vector>
#include <iostream>

using namespace std;

// enviar histograma (256 ints) al destino
void enviarHistogramasAlMaestro(const std::vector<int>& hist, int destino, int tag) {
    MPI_Send((void*)hist.data(), 256, MPI_INT, destino, 20 + tag, MPI_COMM_WORLD);
}

// recibir histogramas desde todos los slaves (1..world_size-1)
std::vector<std::vector<int>> recibirHistogramasDesdeEsclavos(int world_size) {
    vector<vector<int>> recibidos;
    MPI_Status status;
    for (int src = 1; src < world_size; ++src) {
        vector<int> h(256,0);
        MPI_Recv(h.data(), 256, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        recibidos.push_back(h);
        cout << "[MASTER] Recibido histograma desde " << status.MPI_SOURCE << " (tag " << status.MPI_TAG << ")\n";
    }
    return recibidos;
}
