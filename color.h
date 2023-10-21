#pragma once

#include "vec3.h"

#include <iostream>

using color = vec3;

inline void write_color(std::ostream& out, color pixel_color, int samples_per_pixel) {
    auto scaled = pixel_color / samples_per_pixel;
    auto r = scaled.x();
    auto g = scaled.y();
    auto b = scaled.z();
    
    // Write the translated [0,255] value of each color component.
    static const interval intensity(0., 0.999);
    out << static_cast<int>(256 * intensity.clamp(r)) << ' '
        << static_cast<int>(256 * intensity.clamp(g)) << ' '
        << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}
