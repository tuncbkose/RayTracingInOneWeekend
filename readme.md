Going through [Ray Tracing in One Weekend](https://raytracing.github.io/) series.

- I decided to (optionally) implement vectors in SIMD, we'll see if there'll be more opportunities to use it.
  - Controlled by `USE_SIMD` on top of `main.cc`.
- PRNG is provided by [PCG](http://www.pcg-random.org/), because I read somewhere it is good and fast.
  - render time of Book 1 Image 22 (avg, n=10) was shorter when I used `std::default_random_engine` over `pcg32`, so that may not be true (at least for this setting).
- Not sure if I should parallelize now or after book 2 or 3.

Todos:
  - port to CUDA
  - add proper timing measure/image format