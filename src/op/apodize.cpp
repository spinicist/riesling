#include "apodize.hpp"
#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

template<typename Scalar>
ApodizeOp<Scalar>::ApodizeOp(InputDims const &inSize, GridBase<Scalar, 3> *gridder)
{
  std::copy_n(inSize.begin(), 5, sz_.begin());
  for (Index ii = 0; ii < 2; ii++) {
    res_[ii] = 1;
    brd_[ii] = sz_[ii];
  }
  for (Index ii = 2; ii < 5; ii++) {
    res_[ii] = sz_[ii];
    brd_[ii] = 1;
  }
  apo_ = gridder->apodization(LastN<3>(sz_));
}

template<typename Scalar>
auto ApodizeOp<Scalar>::inputDimensions() const -> InputDims
{
  return sz_;
}

template<typename Scalar>
auto ApodizeOp<Scalar>::outputDimensions() const -> OutputDims
{
  return sz_;
}

template<typename Scalar>
auto ApodizeOp<Scalar>::forward(Input const &x) const -> Output
{
  Output result(outputDimensions());
  result.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_).template cast<Scalar>();
  LOG_DEBUG("Apodize Forward Norm {}->{}", Norm(x), Norm(result));
  return result;
}

template<typename Scalar>
auto ApodizeOp<Scalar>::adjoint(Output const &x) const -> Input
{
  Input result(inputDimensions());
  result.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_).template cast<Scalar>();
  LOG_DEBUG("Apodize Adjoint Norm {}->{}", Norm(x), Norm(result));
  return result;
}

template struct ApodizeOp<Cx>;

} // namespace rl
