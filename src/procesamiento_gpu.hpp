#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Declare the kernel launcher as an extern "C" function
extern "C" void aplicarLUTCUDA_Lote(const std::vector<std::string>& rutas, const std::vector<unsigned char>& lut_data);
extern "C" cv::Mat procesarGPU_return_empty();

// We need to declare the kernel wrapper to be able to call it from the test
void aplicarLUTKernel_wrapper(const unsigned char* d_imagen_in, unsigned char* d_imagen_out, const unsigned char* d_lut, int totalPix);
