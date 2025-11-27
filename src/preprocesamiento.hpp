#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

std::vector<unsigned char> calcularLUT(const cv::Mat& img);
std::vector<unsigned char> preprocesarImagenYCalcularLUT(const std::string& ruta);
std::vector<std::string> listarImagenesEnCarpeta(const std::string& carpeta);
std::vector<unsigned char> preprocesarLoteYCalcularLUTs(const std::vector<std::string>& rutas);
