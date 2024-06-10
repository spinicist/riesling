#include "apodize.hpp"
#include "../fft.hpp"
#include "grid.hpp"
#include "log.hpp"
#include "pad.hpp"
#include "tensors.hpp"
#include "threads.hpp"

namespace rl::TOps {

template <typename S, int AD, int OD>
Apodize<S, AD, OD>::Apodize(InDims const ish, Sz<AD> const gshape, std::shared_ptr<Kernel<Scalar, AD>> const &kernel)
  : Parent("Apodize", ish, ish)
{
  for (int ii = 0; ii < OD; ii++) {
    res_[ii] = 1;
    brd_[ii] = ishape[ii];
  }
  for (int ii = OD; ii < AD + OD; ii++) {
    res_[ii] = ishape[ii];
    brd_[ii] = 1;
  }

  auto const            shape = LastN<AD>(this->ishape);
  Eigen::Tensor<Cx, AD> temp(gshape);
  temp.setZero();
  Eigen::Tensor<Cx, AD> k = kernel->at(Eigen::Matrix<float, AD, 1>::Zero()).template cast<Cx>();
  float const           scale = std::sqrt(Product(shape));
  k = k * k.constant(scale);
  TOps::Pad<Cx, AD, AD> padK(k.dimensions(), temp.dimensions());
  temp = padK.forward(k);
  FFT::Adjoint(temp, FFT::PhaseShift(temp.dimensions()));
  TOps::Pad<Cx, AD, AD> padA(shape, gshape);
  apo_.resize(shape);
  apo_.device(Threads::GlobalDevice()) = padA.adjoint(temp).abs().inverse().template cast<Cx>();
}

template <typename S, int AD, int OD> void Apodize<S, AD, OD>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y);
  y.device(Threads::GlobalDevice()) = x * apo_.reshape(res_).broadcast(brd_);
  this->finishForward(y, time);
}

template <typename S, int AD, int OD> void Apodize<S, AD, OD>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x);
  x.device(Threads::GlobalDevice()) = y * apo_.reshape(res_).broadcast(brd_);
  this->finishAdjoint(x, time);
}

template struct Apodize<Cx, 1, 2>;
template struct Apodize<Cx, 1, 3>;
template struct Apodize<Cx, 2, 2>;
template struct Apodize<Cx, 2, 3>;
template struct Apodize<Cx, 3, 2>;
template struct Apodize<Cx, 3, 3>;

} // namespace rl::TOps
