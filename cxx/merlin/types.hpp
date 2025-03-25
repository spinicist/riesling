#pragma once

#include <itkImage.h>
#include <itkVersorRigid3DTransform.h>

namespace merlin {
using ImageType = itk::Image<float, 3>;
using TransformType = itk::VersorRigid3DTransform<double>;
} // namespace merlin
