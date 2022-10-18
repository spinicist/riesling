#include "apodize.hpp"
#include "grid.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <typename Scalar, size_t NDim>
ApodizeOp<Scalar, NDim>::ApodizeOp(InputDims const &dims, GridBase<Scalar, NDim> *gridder)
  : Parent("ApodizeOp", dims, dims)
{
  init(gridder);
}

template <typename Scalar, size_t NDim>
void ApodizeOp<Scalar, NDim>::init(GridBase<Scalar, NDim> *gridder)
{
  for (Index ii = 0; ii < 2; ii++) {
    res_[ii] = 1;
    brd_[ii] = inputDimensions()[ii];
  }
  for (Index ii = 2; ii < NDim + 2; ii++) {
    res_[ii] = inputDimensions()[ii];
    brd_[ii] = 1;
  }
  apo_ = gridder->apodization(LastN<NDim>(inputDimensions()));
}

template <typename Scalar, size_t NDim>
auto ApodizeOp<Scalar, NDim>::forward(InputMap x) const -> OutputMap
{
  auto const time = this->startForward(x);
  x.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_).template cast<Scalar>();
  this->finishForward(x, time);
  return x;
}

template <typename Scalar, size_t NDim>
auto ApodizeOp<Scalar, NDim>::adjoint(OutputMap x) const -> InputMap
{
  auto const time = this->startAdjoint(x);
  x.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_).template cast<Scalar>();
  this->finishForward(x, time);
  return x;
}

template struct ApodizeOp<Cx, 2>;
template struct ApodizeOp<Cx, 3>;

} // namespace rl
