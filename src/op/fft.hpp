#pragma once

#include "operator.hpp"

#include "fft/fft.hpp"
#include "threads.h"
#include "tensorOps.h"

namespace rl {

template <int Rank, int FFTRank = 3>
struct FFTOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;
  using Tensor = typename Eigen::Tensor<Cx, Rank>;

  FFTOp(InputDims const &dims, bool fftZ = true)
    : dims_{dims}
    , ws_{std::make_shared<Tensor>(dims_)}
  {
    if (fftZ) {
      fft3_ = FFT::Make<5, 3>(*ws_);
    } else {
      Cx4 temp(FirstN<4>(dims_));
      fft2_ = FFT::Make<4, 2>(temp);
    }
  }

  FFTOp(std::shared_ptr<Tensor> ws, bool fftZ = true)
    : dims_{ws->dimensions()}
    , ws_{ws}
  {
    if (fftZ) {
      fft3_ = FFT::Make<5, 3>(*ws_);
    } else {
      Cx4 temp(FirstN<4>(dims_));
      fft2_ = FFT::Make<4, 2>(temp);
    }
  }

  InputDims inputDimensions() const
  {
    return dims_;
  }

  OutputDims outputDimensions() const
  {
    return dims_;
  }

  template <typename T>
  Tensor const &forward(T const &x) const
  {
    LOG_DEBUG("Out-of-place Forward FFT Op. Norm {}", Norm(x));
    ws_->device(Threads::GlobalDevice()) = x;
    if (fft3_) {
      fft3_->forward(*ws_);
    } else {
      for (Index iz = 0; iz < ws_->dimension(4); iz++) {
        Cx4 chip = ws_->chip(iz, 4);
        fft2_->forward(chip);
        ws_->chip(iz, 4) = chip;
      }
    }
    Log::Tensor(*ws_, "fft-forward");
    return *ws_;
  }

  template <typename T>
  Tensor &adjoint(T const &x) const
  {
    LOG_DEBUG("Out-of-place Adjoint FFT Op. Norm {}", Norm(x));
    ws_->device(Threads::GlobalDevice()) = x;
    if (fft3_) {
      fft3_->reverse(*ws_);
    } else {
      for (Index iz = 0; iz < ws_->dimension(4); iz++) {
        Cx4 chip = ws_->chip(iz, 4);
        fft2_->reverse(chip);
        ws_->chip(iz, 4) = chip;
      }
    }
    Log::Tensor(*ws_, "fft-adjoint");
    return *ws_;
  }

  Tensor const &forward(Tensor &x) const
  {
    LOG_DEBUG("In-place Forward FFT Op. Norm {}", Norm(x));
    if (fft3_) {
      fft3_->forward(x);
    } else {
      for (Index iz = 0; iz < x.dimension(4); iz++) {
        Cx4 chip = x.chip(iz, 4);
        fft2_->forward(chip);
        x.chip(iz, 4) = chip;
      }
    }
    Log::Tensor(x, "fft-forward");
    return x;
  }

  Tensor &adjoint(Tensor &x) const
  {
    LOG_DEBUG("In-place Adjoint FFT Op. Norm {}", Norm(x));
    if (fft3_) {
      fft3_->reverse(x);
    } else {
      for (Index iz = 0; iz < x.dimension(4); iz++) {
        Cx4 chip = x.chip(iz, 4);
        fft2_->reverse(chip);
        x.chip(iz, 4) = chip;
      }
    }
    Log::Tensor(x, "fft-adjoint");
    return x;
  }

private:
  InputDims dims_;
  std::shared_ptr<Tensor> ws_;
  std::unique_ptr<FFT::FFT<5, 3>> fft3_;
  std::unique_ptr<FFT::FFT<4, 2>> fft2_;
};
} // namespace rl
