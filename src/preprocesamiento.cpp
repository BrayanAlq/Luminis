// preprocesamiento.cpp
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>

using namespace std;
namespace fs = std::filesystem;

// lista de imágenes (utilidad)
std::vector<std::string> listarImagenesEnCarpeta(const std::string& carpeta) {
    vector<string> rutas;
    for (const auto& e : fs::directory_iterator(carpeta))
        if (e.is_regular_file())
            rutas.push_back(e.path().string());
    sort(rutas.begin(), rutas.end());
    return rutas;
}

// calcula histograma de una imagen (serial dentro de hilo)
std::vector<int> preprocesarImagenYCalcularHist(const std::string& ruta) {
    cv::Mat img = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
    vector<int> hist(256, 0);
    if (img.empty()) {
        cerr << "[PRE] Error cargando " << ruta << "\n";
        return hist;
    }
    // conteo simple
    for (int r = 0; r < img.rows; ++r) {
        const uchar* rowptr = img.ptr<uchar>(r);
        for (int c = 0; c < img.cols; ++c) {
            hist[rowptr[c]]++;
        }
    }
    return hist;
}

// preprocesa un lote [inicio,fin) y suma histogramas por nodo
std::vector<int> preprocesarLote(const std::vector<std::string>& rutas, int inicio, int fin) {
    vector<int> histGlobal(256, 0);
    if (inicio >= fin) return histGlobal;

    // procesar cada imagen en paralelo (cada iteración es una imagen)
    #pragma omp parallel
    {
        vector<int> histLocal(256, 0);
        #pragma omp for nowait
        for (int i = inicio; i < fin; ++i) {
            auto h = preprocesarImagenYCalcularHist(rutas[i]);
            for (int k = 0; k < 256; ++k) histLocal[k] += h[k];
        }
        #pragma omp critical
        {
            for (int k = 0; k < 256; ++k) histGlobal[k] += histLocal[k];
        }
    }
    return histGlobal;
}
