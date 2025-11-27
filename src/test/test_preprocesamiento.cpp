#include <gtest/gtest.h>
#include "preprocesamiento.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

TEST(PreprocesamientoTest, CalcularLUT) {
    // Create a simple 4x4 grayscale image with values from 0 to 15
    cv::Mat test_image(4, 4, CV_8UC1);
    std::iota(test_image.data, test_image.data + 16, 0);


    // Calculate the LUT
    std::vector<unsigned char> lut = calcularLUT(test_image);

    // Expected LUT for a linear distribution of pixels
    std::vector<unsigned char> expected_lut(256);
    for(int i = 0; i < 256; ++i) {
        if (i < 16) {
            expected_lut[i] = static_cast<unsigned char>((i / 15.0) * 255.0);
        } else {
            expected_lut[i] = 255;
        }
    }


    // The calculated LUT should be close to the expected one.
    // Due to floating point arithmetic, there might be small differences.
    for(size_t i = 0; i < lut.size(); ++i) {
        ASSERT_NEAR(lut[i], expected_lut[i], 1);
    }
}
