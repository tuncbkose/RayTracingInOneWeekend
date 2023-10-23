#pragma once

// Constants

const float infinity = std::numeric_limits<float>::infinity();

// Utility Functions

static inline int divup(int a, int b){
    // Divide and round up, to calculate number of blocks
    return (a + b - 1)/b;
}

// Common headers

#include "../ray.h"
#include "vec3.h"

