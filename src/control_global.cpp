// control_global.cpp
#include <iostream>
#include <chrono>

using namespace std;
using clk = std::chrono::high_resolution_clock;

static clk::time_point t_start;

void inicializarControl() {
    cout << "[CONTROL] Inicializando sistema híbrido (MPI + OpenMP + CUDA)\n";
    t_start = clk::now();
}

void finalizarControl() {
    auto t_end = clk::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    cout << "[CONTROL] Finalizando. Tiempo total (s): " << elapsed << "\n";
}
