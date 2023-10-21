#pragma once

#include "common.h"

class material;


class hit_record {
    public:
        point3 p;
        vec3 normal;
        shared_ptr<material> mat;
        double t;
        bool front_face; // whether the ray hits from the front (true) or back

        void set_face_normal(const ray& r, const vec3& outward_normal){
            // sets `normal`, assumed `outward_normal` has unit length
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};


class hittable {
    public:
        virtual ~hittable() = default;
        virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};
