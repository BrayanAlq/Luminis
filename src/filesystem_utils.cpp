#include "filesystem_utils.hpp"
#include <filesystem>

namespace fs = std::filesystem;

std::string obtenerNombreSalida(const std::string& rutaEntrada) {
    return "data/output/ecualizado_" + fs::path(rutaEntrada).filename().string();
}

