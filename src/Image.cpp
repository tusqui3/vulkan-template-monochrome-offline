#include "Image.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image(uint32_t width, uint32_t height)
    : width(width), height(height), gray(width * height, 0) {}

void Image::setFromFloat(const float* data) {
    memcpy(gray.data(), data, width * height * sizeof(float));
}

bool Image::exportPNG(const std::string& path) const {

    std::vector<uint8_t> rgb(width*height*3);
    for(uint i = 0; i < gray.size(); i ++){
        uint8_t v = uint8_t(std::clamp(gray[i], 0.f, 1.f) * 255.f + 0.5f);
        rgb[i*3]=rgb[i*3+1]=rgb[i*3+2]=v;
    }

    int success = stbi_write_png(
        path.c_str(),
        static_cast<int>(width),
        static_cast<int>(height),
        3,                  
        rgb.data(),        
        static_cast<int>(width * 3)
    );

    return success != 0;
}
