#pragma once

#include "rl/info.hpp"
#include "rl/types.hpp"

#include <itkImage.h>
#include <itkVersorRigid3DTransform.h>

namespace merlin {
using ImageType = itk::Image<float, 3>;
using TransformType = itk::VersorRigid3DTransform<double>;

auto Import(rl::Re3Map const data, rl::Info const info) -> ImageType::Pointer;
auto ITKToRIESLING(TransformType::Pointer t) -> rl::Transform;
auto Register(ImageType::Pointer fixed, ImageType::Pointer moving, ImageType::Pointer mask) -> TransformType::Pointer;

} // namespace merlin
