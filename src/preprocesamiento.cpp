// preprocesamiento.cpp (FINAL: Implementa el cálculo de LUT con OpenMP)

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cmath> // Para std::round

using namespace std;
namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Utilidad: Lista de imágenes
// -----------------------------------------------------------------------------
std::vector<std::string> listarImagenesEnCarpeta(const std::string& carpeta) {
    vector<string> rutas;
    for (const auto& e : fs::directory_iterator(carpeta))
        if (e.is_regular_file())
            rutas.push_back(e.path().string());
    sort(rutas.begin(), rutas.end());
    return rutas;
}

// -----------------------------------------------------------------------------
// Función Principal del Worker: Calcula Histograma, CDF y LUT
// -----------------------------------------------------------------------------
std::vector<unsigned char> preprocesarImagenYCalcularLUT(const std::string& ruta) {
    cv::Mat img = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
    vector<unsigned char> lut(256);
    
    if (img.empty()) {
        cerr << "[WORKER] Error cargando " << ruta << ". Devolviendo LUT identity.\n";
        for (int i = 0; i < 256; ++i) lut[i] = (unsigned char)i; 
        return lut;
    }
    
    // 1. CÁLCULO DEL HISTOGRAMA
    vector<int> hist(256, 0);
    long long totalPixels = (long long)img.rows * img.cols;
    
    #pragma omp parallel 
    {
        // 1.1. Inicialización local y conteo de píxeles
        vector<int> histLocal(256, 0); 
        
        #pragma omp for nowait
        for (int r = 0; r < img.rows; ++r) {
            const uchar* rowptr = img.ptr<uchar>(r);
            for (int c = 0; c < img.cols; ++c) {
                histLocal[rowptr[c]]++;
            }
        }
        
        // 1.2. Suma final de los histogramas locales (Área crítica de sincronización)
        #pragma omp critical
        {
            for (int k = 0; k < 256; ++k) hist[k] += histLocal[k];
        }
    } // Fin de la región parallel para el Histograma
    
    // 2. CÁLCULO DE LA CDF (Función de Distribución Acumulada)
    vector<long long> cdf(256, 0);
    cdf[0] = hist[0];
    int min_val = (hist[0] > 0) ? 0 : -1; 
    
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
        if (hist[i] > 0 && min_val == -1) {
            min_val = i;
        }
    }
    
    // 3. CÁLCULO DE LA LUT (Tabla de Búsqueda)
    if (totalPixels == 0 || min_val == -1) {
         for (int i = 0; i < 256; ++i) lut[i] = (unsigned char)i;
         return lut;
    }
    
    long long cdf_min = cdf[min_val];
    const double L_MINUS_1 = 255.0;
    const double DENOMINATOR = (double)(totalPixels - cdf_min);

    if (DENOMINATOR == 0.0) {
        for (int i = 0; i < 256; ++i) lut[i] = (unsigned char)i;
        return lut;
    }

    // USO ADICIONAL DE OPENMP: Paralelismo del bucle de 256 iteraciones.
    #pragma omp parallel for
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] == 0) {
            lut[i] = (unsigned char)0; 
        } else {
            double mapped_value = ((double)cdf[i] - cdf_min) / DENOMINATOR * L_MINUS_1;
            lut[i] = (unsigned char)std::min(255, std::max(0, (int)std::round(mapped_value)));
        }
    }
    
    return lut;
}