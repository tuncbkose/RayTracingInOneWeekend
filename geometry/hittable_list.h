#pragma once

#include "hittable.h"

class hittable_list: public hittable {
    public:
        hittable** objects;
        int n_objects;

        __device__ hittable_list() {}
        __device__ hittable_list(hittable** l, int n): objects(l), n_objects(n) {}

        __device__ bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_tmax;

            for (int i=0; i<n_objects; ++i){
                if (objects[i]->hit(r, ray_tmin, closest_so_far, temp_rec)){
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
            return hit_anything;
        }

};

