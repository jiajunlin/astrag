#include <cmath>
#include <vector>

namespace geo {

/// A 2-D vector with the usual algebra.
class Vec2 {
public:
    Vec2(double x, double y) : x_(x), y_(y) {}

    /// Euclidean length of the vector.
    double length() const {
        return std::sqrt(x_ * x_ + y_ * y_);
    }

    /// Dot product with another vector.
    double dot(const Vec2& o) const {
        return x_ * o.x_ + y_ * o.y_;
    }

    double x_, y_;
};

/// Angle in radians between two vectors (0 when parallel).
double angle_between(const Vec2& a, const Vec2& b) {
    double denom = a.length() * b.length();
    if (denom == 0.0) return 0.0;
    return std::acos(a.dot(b) / denom);
}

}  // namespace geo
