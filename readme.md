Going through [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) but
- I decided to (optionally) implement vectors in SIMD, we'll see if there'll be more opportunities to use it.
  - Controlled by `USE_SIMD` on top of `main.cc`.
- PRNG is provided by [PCG](http://www.pcg-random.org/), because I read somewhere it is good and fast.
