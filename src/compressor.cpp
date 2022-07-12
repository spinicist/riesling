#include "compressor.h"

#include "tensorOps.h"

namespace rl {

Index Compressor::out_channels() const
{
  return psi.cols();
}

void Compressor::compress(Cx4 const &source, Cx4 &dest)
{
  assert(source.dimension(1) == dest.dimension(1));
  assert(source.dimension(2) == dest.dimension(2));
  assert(source.dimension(3) == dest.dimension(3));
  assert(source.dimension(0) == psi.rows());
  assert(dest.dimension(0) == psi.cols());
  Log::Print(FMT_STRING("Applying coil compression"));
  auto const sourcemat = CollapseToConstMatrix(source);
  auto destmat = CollapseToMatrix(dest);
  destmat.noalias() = psi.transpose() * sourcemat;
}

Cx3 Compressor::compress(Cx3 const &source)
{
  assert(source.dimension(0) == psi.rows());
  Log::Print(FMT_STRING("Applying coil compression"));
  auto const sourcemat = CollapseToConstMatrix(source);
  Cx3 dest(psi.cols(), source.dimension(1), source.dimension(2));
  auto destmat = CollapseToMatrix(dest);
  destmat.noalias() = psi.transpose() * sourcemat;
  return dest;
}

Cx4 Compressor::compress(Cx4 const &source)
{
  assert(source.dimension(0) == psi.rows());
  Log::Print(FMT_STRING("Applying coil compression"));
  auto const sourcemat = CollapseToConstMatrix(source);
  Cx4 dest(psi.cols(), source.dimension(1), source.dimension(2), source.dimension(3));
  auto destmat = CollapseToMatrix(dest);
  destmat.noalias() = psi.transpose() * sourcemat;
  return dest;
}
} // namespace rl
