#pragma once

#include "hittable.h"
#include "../utils/vec3.h"

class sphere: public hittable {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 _center, float _radius): center(_center), radius(_radius) {}

        __device__ bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const override {
            /*
             *  Returns whether ray r hits a sphere by solving the quadratic equation
             *  (r(t)-center) \cdot (r(t)-center) = radius^2 
             */
            vec3 oc = r.origin() - center;
            auto a = r.direction().squared_length();
            auto half_b = dot(oc, r.direction());
            auto c = oc.squared_length() - radius*radius;
            auto discriminant = half_b*half_b - a*c;

            if (discriminant < 0) return false;
            auto sqrtd = sqrt(discriminant);

            // Find nearest root within range (ray_tmin, ray_tmax)
            auto root = (-half_b - sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root){
                root = (-half_b + sqrtd) / a;
                if (root <= ray_tmin || ray_tmax <= root) return false;
            }

            // record acceptable hit
            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center)/radius;
            rec.set_face_normal(r, outward_normal);
            return true;
        }

    private:
        vec3 center;
        float radius;
};
