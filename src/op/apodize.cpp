#include "apodize.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "grid.hpp"

namespace rl {

template<typename Scalar, size_t NDim>
ApodizeOp<Scalar, NDim>::ApodizeOp(InputDims const &inSize, GridBase<Scalar, NDim> *gridder)
{
  std::copy_n(inSize.begin(), NDim + 2, sz_.begin());
  for (Index ii = 0; ii < 2; ii++) {
    res_[ii] = 1;
    brd_[ii] = sz_[ii];
  }
  for (Index ii = 2; ii < NDim + 2; ii++) {
    res_[ii] = sz_[ii];
    brd_[ii] = 1;
  }
  apo_ = gridder->apodization(LastN<NDim>(sz_));
  x_.resize(inputDimensions());
  y_.resize(outputDimensions());
}

template<typename Scalar, size_t NDim>
auto ApodizeOp<Scalar, NDim>::inputDimensions() const -> InputDims
{
  return sz_;
}

template<typename Scalar, size_t NDim>
auto ApodizeOp<Scalar, NDim>::outputDimensions() const -> OutputDims
{
  return sz_;
}

template<typename Scalar, size_t NDim>
auto ApodizeOp<Scalar, NDim>::forward(Input const &x) const -> Output const &
{
  y_ = x * apo_.reshape(res_).broadcast(brd_).template cast<Scalar>();
  LOG_DEBUG("Apodize Forward Norm {}->{}", Norm(x), Norm(y_));
  return y_;
}

template<typename Scalar, size_t NDim>
auto ApodizeOp<Scalar, NDim>::adjoint(Output const &x) const -> Input const &
{
  x_ = x * apo_.reshape(res_).broadcast(brd_).template cast<Scalar>();
  LOG_DEBUG("Apodize Adjoint Norm {}->{}", Norm(x), Norm(x_));
  return x_;
}

template struct ApodizeOp<Cx, 2>;
template struct ApodizeOp<Cx, 3>;

} // namespace rl
