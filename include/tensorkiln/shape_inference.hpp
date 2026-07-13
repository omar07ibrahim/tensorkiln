#pragma once

#include "tensorkiln/shape.hpp"

namespace tensorkiln {

[[nodiscard]] Result<Shape> infer_broadcast_shape(
    const Shape& left, const Shape& right,
    ShapeLimits limits = ShapeLimits{});

[[nodiscard]] Result<Shape> infer_matmul_shape(
    const Shape& left, const Shape& right,
    ShapeLimits limits = ShapeLimits{});

}  // namespace tensorkiln
