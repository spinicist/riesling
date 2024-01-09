#include "svd.hpp"

#include "algo/decomp.hpp"
#include "log.hpp"

namespace rl {

SVDBasis::SVDBasis(Eigen::ArrayXXf const &dynamics,
                   Index const            nRetain,
                   bool const             demean,
                   bool const             rotate,
                   bool const             normalize,
                   Index const            segSize)
{
  // Calculate SVD - observations are in cols
  Eigen::ArrayXXf d = normalize ? dynamics.colwise().normalized().transpose().eval() : dynamics.transpose().eval();
  if (demean) { d = d.rowwise() - d.colwise().mean(); }
  Index const segs = segSize > 0 ? dynamics.cols() / segSize : 1;
  Index const sz = dynamics.cols() / segs;
  if (segs * sz != dynamics.cols()) {
    Log::Fail("Segment size {} does not cleanly divide dynamic length {}", segSize, dynamics.cols());
  }
  basis.resize(nRetain, dynamics.cols());
  Index st = 0;
  for (Index is = 0; is < segs; is++) {
    auto const svd = SVD<float>(d.middleRows(st, sz));
    Log::Print("Retaining {} basis vectors, variance {}", nRetain, svd.S.head(nRetain).square().sum());
    basis.middleCols(st, sz) = rotate ? svd.equalized(nRetain).transpose().eval() : svd.V.leftCols(nRetain).transpose().eval();
    basis.middleCols(st, sz) *= std::sqrt(basis.cols());
    st += sz;
  }

  Log::Print<Log::Level::Debug>("Orthogonality check:\n{}", fmt::streamed(basis * basis.adjoint()));
}

} // namespace rl
