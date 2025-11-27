#include <mpi.h>
#include <iostream>
#include "control_global.hpp"
#include "gestor_distribucion.hpp"

int modo_mpi_openmp_cuda(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    inicializarControl();

    if (rank == 0) {
        ejecutarMaestro(world_size);
    } else {
        ejecutarEsclavo(rank);
    }

    finalizarControl();
    MPI_Finalize();
    return 0;
}
