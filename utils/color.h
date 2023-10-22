#pragma once

#include "vec3.h"

#include <iostream>

using color = vec3;

inline double linear_to_gamma(double linear_component){
    return sqrt(linear_component); 
}

inline void write_color(std::ostream& out, color pixel_color, int samples_per_pixel) {
    auto scaled = pixel_color / samples_per_pixel;
    auto r = scaled.x();
    auto g = scaled.y();
    auto b = scaled.z();

    // Apply gamma compression
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);
    
    // Write the translated [0,255] value of each color component.
    static const interval intensity(0., 0.999);
    out << static_cast<int>(256 * intensity.clamp(r)) << ' '
        << static_cast<int>(256 * intensity.clamp(g)) << ' '
        << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}
