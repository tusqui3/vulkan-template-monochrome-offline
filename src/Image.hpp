#pragma once
#include <cstdint>
#include <string>
#include <vector>

class Image {
public:
    Image(uint32_t width, uint32_t height);

    void setFromFloat(const float* data);
    bool exportPNG(const std::string& path) const;

    uint32_t         width, height;
    std::vector<float> gray;
};