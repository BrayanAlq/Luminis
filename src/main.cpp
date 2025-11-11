#include <iostream>
#include <string>
#include <cstdlib> // Para atoi
#include "modo_serial.hpp"
#include "modo_openmp_cuda.hpp"
#include "modo_mpi_openmp_cuda.hpp"

using namespace std;

// -----------------------------------------------------------------------------
// Inicialización y finalización global del sistema
// -----------------------------------------------------------------------------
void inicializar_sistema()
{
  cout << "🔧 Inicializando sistema híbrido..." << endl;
}

void finalizar_sistema()
{
  cout << "✅ Finalizando sistema y liberando recursos..." << endl;
}

// -----------------------------------------------------------------------------
// Función para leer el modo de ejecución desde argumentos
// -----------------------------------------------------------------------------
int leer_modo_desde_argumentos(int argc, char **argv)
{
  if (argc < 2)
  {
    cout << "⚠️  Uso: " << argv[0] << " <modo>\n"
         << "   0 = Serial\n"
         << "   1 = OpenMP\n"
         << "   2 = Híbrido (MPI + OpenMP + CUDA)\n";
    exit(1);
  }
  return atoi(argv[1]);
}

// -----------------------------------------------------------------------------
// Función principal
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  inicializar_sistema();

  int modo = leer_modo_desde_argumentos(argc, argv);

  switch (modo)
  {
  case 0:
    cout << "\n🧩 Ejecutando en modo SERIAL..." << endl;
    ejecutarModoSerial();
    break;

  case 1:
    cout << "\n🧩 Ejecutando en modo OPENMP..." << endl;
    ejecutarModoOpenMPCUDA();
    break;

  case 2:
    cout << "\n🧩 Ejecutando en modo HÍBRIDO (MPI + OpenMP + CUDA)..." << endl;
    modo_mpi_openmp_cuda(argc, argv);
    break;

  default:
    cout << "❌ Modo no válido. Usa 0, 1 o 2." << endl;
    break;
  }

  finalizar_sistema();
  return 0;
}
