#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;
namespace fs = std::filesystem;
// ===============================
//   KERNEL CUDA CON LUT
// ===============================
__global__ void aplicarLUTKernel(const unsigned char* img, unsigned char* salida, const unsigned char* LUT, int totalPix)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalPix)
        salida[i] = LUT[img[i]];
}

// ===============================
//   FUNCIÓN CUDA CON LUT
// ===============================
cv::Mat ecualizarHistogramaCUDA(const cv::Mat &imgGray)
{
    int totalPix = imgGray.rows * imgGray.cols;
    vector<int> hist(256, 0);

    // Calcular histograma (CPU)
    for (int y = 0; y < imgGray.rows; y++)
        for (int x = 0; x < imgGray.cols; x++)
            hist[imgGray.at<uchar>(y, x)]++;

    // Calcular CDF
    vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++)
        cdf[i] = cdf[i - 1] + hist[i];

    int total = cdf[255];
    int cdf_min = 0;
    for (int i = 0; i < 256; i++)
        if (cdf[i] > 0) { cdf_min = cdf[i]; break; }

    // Generar LUT (Look-Up Table)
    vector<unsigned char> LUT(256);
    for (int i = 0; i < 256; i++)
    {
        float val = ((float)cdf[i] - cdf_min) / (float)(total - cdf_min);
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        LUT[i] = static_cast<unsigned char>(val * 255.0f);
    }

    // Memoria en GPU
    unsigned char *d_img, *d_out, *d_LUT;
    cudaMalloc(&d_img, totalPix);
    cudaMalloc(&d_out, totalPix);
    cudaMalloc(&d_LUT, 256);

    // Copiar datos
    cudaMemcpy(d_img, imgGray.data, totalPix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LUT, LUT.data(), 256, cudaMemcpyHostToDevice);

    // Kernel
    int blockSize = 256;
    int numBlocks = (totalPix + blockSize - 1) / blockSize;
    aplicarLUTKernel<<<numBlocks, blockSize>>>(d_img, d_out, d_LUT, totalPix);
    cudaDeviceSynchronize();

    // Copiar resultado
    cv::Mat salida(imgGray.rows, imgGray.cols, CV_8UC1);
    cudaMemcpy(salida.data, d_out, totalPix, cudaMemcpyDeviceToHost);

    // Liberar
    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_LUT);

    return salida;
}

// ===============================
//   MODO OPENMP + CUDA (PARALELO)
// ===============================
int ejecutarModoOpenMPCUDA()
{
    string rutaEntrada = "imagenes_entrada/";
    string rutaSalida = "imagenes_salida/";
    fs::create_directories(rutaSalida);

    vector<string> archivos;
    for (const auto &entry : fs::directory_iterator(rutaEntrada))
            if (fs::is_regular_file(entry.path()))
            archivos.push_back(entry.path().string());

    cout << "=== Ecualización de Histograma (OpenMP + CUDA con LUT) ===" << endl;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)archivos.size(); i++)
    {
        string nombreArchivo = fs::path(archivos[i]).filename().string();
        cv::Mat imagen = cv::imread(archivos[i], cv::IMREAD_GRAYSCALE);
        if (imagen.empty())
        {
            #pragma omp critical
            cerr << "Error al cargar: " << nombreArchivo << endl;
            continue;
        }

        cout << "Procesando en hilo " << omp_get_thread_num() << ": " << nombreArchivo << endl;
        cv::Mat resultado = ecualizarHistogramaCUDA(imagen);

        string rutaSalidaFinal = rutaSalida + "EQ_CUDA_" + nombreArchivo;
        cv::imwrite(rutaSalidaFinal, resultado);
    }

    cout << "Proceso híbrido (OpenMP+CUDA) completado." << endl;
    return 0;
}

int main() {
    return ejecutarModoOpenMPCUDA();
}
