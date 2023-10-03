#include "apodize.hpp"
#include "fft/fft.hpp"
#include "grid.hpp"
#include "log.hpp"
#include "pad.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <typename S, int NDim>
ApodizeOp<S, NDim>::ApodizeOp(InDims const                                     ishape,
                              Sz<NDim> const                                   gshape,
                              std::shared_ptr<Kernel<Scalar, NDim>> const &kernel)
  : Parent("ApodizeOp", ishape, ishape)
{
  for (int ii = 0; ii < 2; ii++) {
    res_[ii] = 1;
    brd_[ii] = ishape[ii];
  }
  for (int ii = 2; ii < NDim + 2; ii++) {
    res_[ii] = ishape[ii];
    brd_[ii] = 1;
  }

  auto const              shape = LastN<NDim>(this->ishape);
  Eigen::Tensor<Cx, NDim> temp(gshape);
  auto const              fft = FFT::Make<NDim, NDim>(temp);
  temp.setZero();
  Eigen::Tensor<Cx, NDim> k = kernel->at(Eigen::Matrix<float, NDim, 1>::Zero()).template cast<Cx>();
  float const             scale = std::sqrt(Product(shape));
  k = k * k.constant(scale);
  PadOp<Cx, NDim, NDim> padK(k.dimensions(), temp.dimensions());
  temp = padK.forward(k);
  fft->reverse(temp);
  PadOp<Cx, NDim, NDim> padA(shape, gshape);
  apo_.resize(shape);
  apo_.device(Threads::GlobalDevice()) = padA.adjoint(temp).abs().inverse().template cast<Cx>();
  LOG_DEBUG("Apodization size {} Scale: {} Norm: {}", shape, scale, Norm(apo_));
}

template <typename S, int NDim>
void ApodizeOp<S, NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  y.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_);
  this->finishForward(y, time);
}

template <typename S, int NDim>
void ApodizeOp<S, NDim>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  x.device(Threads::GlobalDevice()) = y * apo_.reshape(res_).broadcast(brd_);
  this->finishForward(x, time);
}

template struct ApodizeOp<Cx, 1>;
template struct ApodizeOp<Cx, 2>;
template struct ApodizeOp<Cx, 3>;

} // namespace rl
