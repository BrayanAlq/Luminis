// preprocesamiento.cpp (MODIFICADO para flujo MPI/LUT)

#include <opencv2/opencv.hpp>
#include <omp.h> // Aunque no se usa directamente aquí, se mantiene para futuro uso en preprocesamiento
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <algorithm> // Para std::min y std::max

using namespace std;
namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Utilidad: Lista de imágenes
// -----------------------------------------------------------------------------
std::vector<std::string> listarImagenesEnCarpeta(const std::string& carpeta) {
    // ... (El código de esta función permanece igual) ...
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
// Retorna la Tabla de Búsqueda (LUT) de 256 elementos (uchar)
std::vector<unsigned char> preprocesarImagenYCalcularLUT(const std::string& ruta) {
    cv::Mat img = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
    vector<unsigned char> lut(256); // El resultado final a devolver
    
    if (img.empty()) {
        cerr << "[WORKER] Error cargando " << ruta << ". Devolviendo LUT vacía.\n";
        // Devuelve una LUT identity simple o un patrón si falla
        for (int i = 0; i < 256; ++i) lut[i] = (unsigned char)i; 
        return lut;
    }
    
    // 1. CÁLCULO DEL HISTOGRAMA (Adaptado de tu código anterior)
    vector<int> hist(256, 0);
    long long totalPixels = img.rows * img.cols; // Usar long long para evitar overflow
    
    for (int r = 0; r < img.rows; ++r) {
        const uchar* rowptr = img.ptr<uchar>(r);
        for (int c = 0; c < img.cols; ++c) {
            hist[rowptr[c]]++;
        }
    }
    
    // 2. CÁLCULO DE LA CDF (Función de Distribución Acumulada)
    vector<long long> cdf(256, 0);
    cdf[0] = hist[0];
    int min_val = hist[0] > 0 ? 0 : -1; // Encuentra el primer nivel de gris no nulo
    
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
        if (hist[i] > 0 && min_val == -1) {
            min_val = i;
        }
    }
    
    // 3. CÁLCULO DE LA LUT (Tabla de Búsqueda)
    long long cdf_min = cdf[min_val];
    
    if (totalPixels == 0 || cdf_min == cdf[255]) {
         // Si la imagen es uniforme, devuelve la LUT identity (no hay ecualización)
         for (int i = 0; i < 256; ++i) lut[i] = (unsigned char)i;
         return lut;
    }

    // Fórmula de Ecualización: T(i) = (CDF(i) - CDF_min) / (Total - CDF_min) * (L-1)
    const double L_MINUS_1 = 255.0;
    const double DENOMINATOR = (double)(totalPixels - cdf_min);

    for (int i = 0; i < 256; ++i) {
        if (cdf[i] == 0) {
            // Si el nivel de gris no existe, mapea a 0 o al nivel anterior
            lut[i] = (unsigned char)0; 
        } else {
            // Aplica la fórmula
            double mapped_value = ((double)cdf[i] - cdf_min) / DENOMINATOR * L_MINUS_1;
            // Asegura que el valor esté entre 0 y 255
            lut[i] = (unsigned char)std::min(255, std::max(0, (int)std::round(mapped_value)));
        }
    }
    
    return lut;
}

// Nota: preprocesarLote ha sido eliminado, ya que el Worker ahora procesa 
// las imágenes de una en una gracias al Task Farming de MPI.