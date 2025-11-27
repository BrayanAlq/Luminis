#include <gtest/gtest.h>
#include "filesystem_utils.hpp"

TEST(FilesystemUtilsTest, ObtenerNombreSalida) {
    // Test case 1: Simple path
    std::string input1 = "path/to/file.txt";
    std::string expected1 = "file_output.txt";
    ASSERT_EQ(obtenerNombreSalida(input1), expected1);

    // Test case 2: Path with no extension
    std::string input2 = "path/to/file";
    std::string expected2 = "file_output";
    ASSERT_EQ(obtenerNombreSalida(input2), expected2);

    // Test case 3: Path with multiple dots
    std::string input3 = "path/to/file.name.with.dots.txt";
    std::string expected3 = "file.name.with.dots_output.txt";
    ASSERT_EQ(obtenerNombreSalida(input3), expected3);

    // Test case 4: Empty path
    std::string input4 = "";
    std::string expected4 = "_output";
    ASSERT_EQ(obtenerNombreSalida(input4), expected4);
}
