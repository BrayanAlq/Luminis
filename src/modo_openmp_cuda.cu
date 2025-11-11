
using namespace std;
namespace fs = std::filesystem;

__global__ void ecualizarKernel(unsigned char* img, unsigned char* salida, int totalPix, int* cdf, int cdf_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalPix)
    {
        int valor = img[i];
        float nuevoValor = ((float)cdf[valor] - cdf_min) / (float)(cdf[255] - cdf_min) * 255.0f;
        salida[i] = (unsigned char)max(0.0f, min(255.0f, nuevoValor));
    }
}

cv::Mat ecualizarHistogramaCUDA(const cv::Mat &imgGray)
{
    int totalPix = imgGray.rows * imgGray.cols;
    vector<int> hist(256, 0);

    // Cálculo del histograma (en CPU)
    for (int y = 0; y < imgGray.rows; y++)
        for (int x = 0; x < imgGray.cols; x++)
            hist[imgGray.at<uchar>(y, x)]++;

    // Calcular CDF (CPU)
    vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++)
        cdf[i] = cdf[i - 1] + hist[i];
    int cdf_min = 0;
    for (int i = 0; i < 256; i++)
        if (cdf[i] > 0) { cdf_min = cdf[i]; break; }

    // Copiar datos a GPU
    unsigned char *d_img, *d_out;
    int *d_cdf;
    cudaMalloc(&d_img, totalPix);
    cudaMalloc(&d_out, totalPix);
    cudaMalloc(&d_cdf, 256 * sizeof(int));

    cudaMemcpy(d_img, imgGray.data, totalPix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cdf, cdf.data(), 256 * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (totalPix + blockSize - 1) / blockSize;

    ecualizarKernel<<<numBlocks, blockSize>>>(d_img, d_out, totalPix, d_cdf, cdf_min);
    cudaDeviceSynchronize();

    cv::Mat salida(imgGray.rows, imgGray.cols, CV_8UC1);
    cudaMemcpy(salida.data, d_out, totalPix, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_cdf);

    return salida;
}

int ejecutarModoOpenMPCUDA()
{
    string rutaEntrada = "imagenes_entrada/";
    string rutaSalida = "imagenes_salida/";
    fs::create_directories(rutaSalida);

    vector<string> archivos;
    for (const auto &entry : fs::directory_iterator(rutaEntrada))
        if (entry.is_regular_file())
            archivos.push_back(entry.path().string());

    cout << "=== Ecualización Híbrida (OpenMP + CUDA) ===" << endl;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)archivos.size(); i++)
    {
        string nombreArchivo = fs::path(archivos[i]).filename().string();
        cv::Mat imagen = cv::imread(archivos[i], cv::IMREAD_GRAYSCALE);
        if (imagen.empty())
        {
            cerr << "Error al cargar: " << nombreArchivo << endl;
            continue;
        }

        cout << "Procesando (hilo " << omp_get_thread_num() << "): " << nombreArchivo << endl;
        cv::Mat resultado = ecualizarHistogramaCUDA(imagen);

        string rutaSalidaFinal = rutaSalida + "EQ_CUDA_" + nombreArchivo;
        cv::imwrite(rutaSalidaFinal, resultado);
    }

    cout << "Proceso híbrido completado." << endl;
    return 0;
}
