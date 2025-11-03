// procesamiento_gpu.cu
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

// Ejemplo simple: calculamos CDF en host y mostramos (placeholder)
// En implementación completa aquí copiarías CDF a device, ejecutarías kernel que aplica LUT sobre la imagen
extern "C" void postprocesar_en_GPU(const std::vector<int>& histGlobal) {
    // calcular CDF (host)
    int cdf[256];
    cdf[0] = histGlobal[0];
    for (int i = 1; i < 256; ++i) cdf[i] = cdf[i-1] + histGlobal[i];
    int total = cdf[255];
    cout << "[GPU] CDF calculada (total pixeles = " << total << ")\n";

    // aquí normalmente copiarías cdf a GPU y aplicarías kernel que utilice una LUT para mapear la imagen.
    // Por ahora solo imprimimos un resumen:
    int nonzero_min = 0;
    while (nonzero_min < 256 && cdf[nonzero_min] == 0) ++nonzero_min;
    cout << "[GPU] Primer bin con pixeles: " << nonzero_min << "\n";
}

// placeholder que devuelve cv::Mat vacío (en versión completa se retornaría la imagen ecualizada)
extern "C" cv::Mat procesarGPU_return_empty() {
    return cv::Mat();
}
