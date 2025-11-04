#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

cv::Mat ecualizarHistogramaGrises(const cv::Mat& imgGray) {
    vector<int> hist(256, 0);
    for (int y = 0; y < imgGray.rows; y++)
        for (int x = 0; x < imgGray.cols; x++)
            hist[imgGray.at<uchar>(y, x)]++;

    vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++)
        cdf[i] = cdf[i - 1] + hist[i];

    int totalPixeles = imgGray.rows * imgGray.cols;
    float denom = (float)(totalPixeles - cdf[0]);
    if (denom == 0) denom = 1; // evitar división por cero

    cv::Mat ecualizada = imgGray.clone();
    for (int y = 0; y < imgGray.rows; y++) {
        for (int x = 0; x < imgGray.cols; x++) {
            int valor = imgGray.at<uchar>(y, x);
            uchar nuevoValor = (uchar)(((float)cdf[valor] - cdf[0]) / denom * 255.0f);
            ecualizada.at<uchar>(y, x) = nuevoValor;
        }
    }

    return ecualizada;
}

int main() {
    string rutaEntrada = "imagenes_entrada/";
    string rutaSalida = "imagenes_salida/";
    fs::create_directories(rutaSalida);

    if (!fs::exists(rutaEntrada)) {
        cerr << "La carpeta de entrada no existe: " << rutaEntrada << endl;
        return -1;
    }

    cout << "=== Ecualización de Histograma (Versión Serial) ===" << endl;

    for (const auto& entry : fs::directory_iterator(rutaEntrada)) {
        if (entry.is_regular_file()) {
            string nombreArchivo = entry.path().filename().string();
            cv::Mat imagen = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

            if (imagen.empty()) {
                cerr << "Error al cargar: " << nombreArchivo << endl;
                continue;
            }

            cout << "Procesando: " << nombreArchivo << " ..." << endl;

            cv::Mat imagenEcualizada = ecualizarHistogramaGrises(imagen);

            string rutaSalidaFinal = rutaSalida + "EQ_" + nombreArchivo;
            cv::imwrite(rutaSalidaFinal, imagenEcualizada);

            cout << "Guardado en: " << rutaSalidaFinal << endl;
        }
    }

    cout << "Proceso completado. Imágenes guardadas en: " << rutaSalida << endl;
    return 0;
}

