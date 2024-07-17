#include "fft.hpp"
#include "../fft.hpp"

#include <fmt/format.h>

namespace rl::TOps {

template <int Rank, int FFTRank>
FFT<Rank, FFTRank>::FFT(InDims const &dims, bool const adj)
  : Parent(fmt::format("FFT{}", adj ? " Inverse" : ""), dims, dims)
  , adjoint_{adj}
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
  ph_ = rl::FFT::PhaseShift(LastN<FFTRank>(ishape));
}

template <int Rank, int FFTRank>
FFT<Rank, FFTRank>::FFT(InMap x)
  : Parent("FFT", x.dimensions(), x.dimensions())
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
}

template <int Rank, int FFTRank>
auto FFT<Rank, FFTRank>::inverse() const -> std::shared_ptr<rl::Ops::Op<Cx>>
{
  return std::make_shared<FFT<Rank, FFTRank>>(this->ishape, !this->adjoint_);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y = x;
  if (adjoint_) {
    rl::FFT::Adjoint(y, dims_, ph_);
  } else {
    rl::FFT::Forward(y, dims_, ph_);
  }
  this->finishForward(y, time, false);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x = y;
  if (adjoint_) {
    rl::FFT::Forward(x, dims_, ph_);
  } else {
    rl::FFT::Adjoint(x, dims_, ph_);
  }
  this->finishAdjoint(x, time, false);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  InTensor   tmp = x;
  if (adjoint_) {
    rl::FFT::Adjoint(tmp, dims_, ph_);
  } else {
    rl::FFT::Forward(tmp, dims_, ph_);
  }
  y += tmp;
  this->finishForward(y, time, true);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  InTensor   tmp = y;
  if (adjoint_) {
    rl::FFT::Forward(tmp, dims_, ph_);
  } else {
    rl::FFT::Adjoint(tmp, dims_, ph_);
  }
  x += tmp;
  this->finishAdjoint(x, time, true);
}

template struct FFT<4, 3>;
template struct FFT<5, 3>;

} // namespace rl::TOps
