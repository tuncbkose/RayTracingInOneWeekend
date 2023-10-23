#pragma once

#include "../ray.h"

// Class for recording rays hitting objects
class hit_record {
    public:
        vec3 p;  // point of hit
        vec3 normal;
        float t; // R(t) = orig + t*dir = p
        bool front_face;

        __device__ void set_face_normal(const ray& r, const vec3& outward_normal){
            // Sets the normal vector to face the incident ray
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};


// ABC for hittable objects
class hittable {
    public:
        __device__ virtual bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const = 0;
};

