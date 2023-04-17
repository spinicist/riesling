#include "apodize.hpp"
#include "grid.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <size_t NDim>
ApodizeOp<NDim>::ApodizeOp(InDims const &dims, GridBase<Scalar, NDim> *gridder)
  : Parent("ApodizeOp", dims, dims)
{
  init(gridder);
}

template <size_t NDim>
void ApodizeOp<NDim>::init(GridBase<Scalar, NDim> *gridder)
{
  for (size_t ii = 0; ii < 2; ii++) {
    res_[ii] = 1;
    brd_[ii] = ishape[ii];
  }
  for (size_t ii = 2; ii < NDim + 2; ii++) {
    res_[ii] = ishape[ii];
    brd_[ii] = 1;
  }
  apo_ = gridder->apodization(LastN<NDim>(ishape)).template cast<Cx>();
}

template <size_t NDim>
void ApodizeOp<NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  y.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_);
  this->finishForward(y, time);
}

template <size_t NDim>
void ApodizeOp<NDim>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  x.device(Threads::GlobalDevice()) = y * apo_.reshape(res_).broadcast(brd_);
  this->finishForward(x, time);
}

template struct ApodizeOp<1>;
template struct ApodizeOp<2>;
template struct ApodizeOp<3>;

} // namespace rl
