#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <random>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

inline double random_double(){
    static thread_local std::default_random_engine e;
    std::uniform_real_distribution<double> dis;
    return dis(e);
}

inline double random_double(double min, double max){
    // Returns a random real in [min,max)
    return min + (max-min)*random_double();
}

// Common Headers

#include "../ray.h"
#include "interval.h"
#include "vec3.h"
