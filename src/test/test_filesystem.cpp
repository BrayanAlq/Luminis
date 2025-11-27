#include <gtest/gtest.h>
#include "filesystem_utils.hpp"

TEST(FilesystemUtilsTest, ObtenerNombreSalida) {
    // Test case 1: Ruta simple
    std::string input1 = "path/to/file.txt";
    std::string expected1 = "data/output/ecualizado_file.txt";
    ASSERT_EQ(obtenerNombreSalida(input1), expected1);

    // Test case 2: Ruta sin extensión
    std::string input2 = "path/to/file";
    std::string expected2 = "data/output/ecualizado_file";
    ASSERT_EQ(obtenerNombreSalida(input2), expected2);

    // Test case 3: Ruta con múltiples puntos
    std::string input3 = "path/to/file.name.with.dots.txt";
    std::string expected3 = "data/output/ecualizado_file.name.with.dots.txt";
    ASSERT_EQ(obtenerNombreSalida(input3), expected3);

    // Test case 4: Ruta vacía
    std::string input4 = "";
    std::string expected4 = "data/output/ecualizado_";
    ASSERT_EQ(obtenerNombreSalida(input4), expected4);
}
