#include "apodize.hpp"
#include "grid.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <size_t NDim>
ApodizeOp<NDim>::ApodizeOp(InputDims const &dims, GridBase<Scalar, NDim> *gridder)
  : Parent("ApodizeOp", dims, dims)
{
  init(gridder);
}

template <size_t NDim>
void ApodizeOp<NDim>::init(GridBase<Scalar, NDim> *gridder)
{
  for (size_t ii = 0; ii < 2; ii++) {
    res_[ii] = 1;
    brd_[ii] = inputDimensions()[ii];
  }
  for (size_t ii = 2; ii < NDim + 2; ii++) {
    res_[ii] = inputDimensions()[ii];
    brd_[ii] = 1;
  }
  apo_ = gridder->apodization(LastN<NDim>(inputDimensions())).template cast<Cx>();
}

template <size_t NDim>
auto ApodizeOp<NDim>::forward(InputMap x) const -> OutputMap
{
  auto const time = this->startForward(x);
  x.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_);
  this->finishForward(x, time);
  return x;
}

template <size_t NDim>
auto ApodizeOp<NDim>::adjoint(OutputMap x) const -> InputMap
{
  auto const time = this->startAdjoint(x);
  x.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_);
  this->finishForward(x, time);
  return x;
}

template struct ApodizeOp<1>;
template struct ApodizeOp<2>;
template struct ApodizeOp<3>;

} // namespace rl
