#include <math.h>
#include "mathx.h"

/* Greatest common divisor via Euclid's algorithm.
 * Always returns a non-negative value. */
int gcd(int a, int b) {
    a = abs(a);
    b = abs(b);
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

/* Clamp `v` into the inclusive range [lo, hi]. */
float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

struct Interval {
    float lo;
    float hi;
};
