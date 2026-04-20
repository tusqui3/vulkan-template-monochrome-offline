#include <iostream>
#include "Renderer.hpp"

int main() {
    try {
        Renderer renderer(800, 600);
        Image img = renderer.render();

        if (img.exportPNG("output.png"))
            std::cout << "Saved: output.png (" << img.width << "x" << img.height << ")\n";
        else
            std::cerr << "Failed to write output.png\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
