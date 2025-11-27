#include <gtest/gtest.h>
#include "preprocesamiento.hpp"
#include <opencv2/opencv.hpp>

TEST(PreprocesamientoTest, CalcularLUT) {
    // 1. Crear una imagen simple de 4x4 (16 píxeles totales)
    // Valores distribuidos uniformemente: 0, 16, 32, ..., 240
    cv::Mat test_image(4, 4, CV_8UC1);
    
    for (int i = 0; i < 16; ++i) {
        test_image.data[i] = i * 16;
    }

    // 2. Calcular la LUT
    std::vector<unsigned char> lut = calcularLUT(test_image);

    // 3. Verificaciones Específicas
    // Verificamos que la LUT transforme correctamente los valores QUE EXISTEN en la imagen.
    
    // Cálculo manual para verificar:
    // Total píxeles = 16. Min CDF (cdf_min) = 1 (para el valor 0).
    // Denominador = 16 - 1 = 15. Factor = 255.
    
    // Caso valor 0: cdf = 1. Formula: (1 - 1)/15 * 255 = 0.
    ASSERT_NEAR(lut[0], 0, 1);

    // Caso valor 16 (índice 1 de la imagen): cdf = 2. Formula: (2 - 1)/15 * 255 = 17.
    ASSERT_NEAR(lut[16], 17, 1);

    // Caso valor 32 (índice 2): cdf = 3. Formula: (3 - 1)/15 * 255 = 34.
    ASSERT_NEAR(lut[32], 34, 1);

    // Caso valor 240 (último): cdf = 16. Formula: (16 - 1)/15 * 255 = 255.
    ASSERT_NEAR(lut[240], 255, 1);

    // 4. Verificación de integridad de pasos intermedios
    // Dado que no hay píxeles con valor 1, lut[1] debe ser igual a lut[0]
    ASSERT_EQ(lut[1], lut[0]);
}
