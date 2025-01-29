#include "compressor.hpp"

#include "tensors.hpp"
#include "sys/threads.hpp"

namespace rl {

Index Compressor::out_channels() const { return psi.cols(); }

Cx4 Compressor::compress(Cx4 const &source)
{
  Log::Print("CC", "Compressing to {} channels", psi.cols());
  auto const sourcemat = CollapseToConstMatrix(source);
  if (sourcemat.rows() != psi.rows()) {
    throw Log::Failure("CC", "Number of channels in data {} does not match compression matrix {}", sourcemat.rows(), psi.rows());
  }
  Cx4        dest(psi.cols(), source.dimension(1), source.dimension(2), source.dimension(3));
  auto       destmat = CollapseToMatrix(dest);
  destmat.device(Threads::CoreDevice()).noalias() = psi.transpose() * sourcemat;
  return dest;
}

Cx5 Compressor::compress(Cx5 const &source)
{
  Log::Print("CC", "Compressing to {} channels", psi.cols());
  auto const sourcemat = CollapseToConstMatrix(source);
  if (sourcemat.rows() != psi.rows()) {
    throw Log::Failure("CC", "Number of channels in data {} does not match compression matrix {}", sourcemat.rows(), psi.rows());
  }
  Cx5        dest(psi.cols(), source.dimension(1), source.dimension(2), source.dimension(3), source.dimension(4));
  auto       destmat = CollapseToMatrix(dest);
  destmat.device(Threads::CoreDevice()).noalias() = psi.transpose() * sourcemat;
  return dest;
}
} // namespace rl
