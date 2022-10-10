#pragma once

#include "operator.hpp"

#include "fft/fft.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <int Rank, int FFTRank>
struct FFTOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;
  using Tensor = typename Eigen::Tensor<Cx, Rank>;

  FFTOp(InputDims const &dims)
    : dims_{dims}
    , ws_{std::make_shared<Tensor>(dims_)}, fft_{FFT::Make<Rank, FFTRank>(*ws_)}
  {
  }

  FFTOp(std::shared_ptr<Tensor> ws)
    : dims_{ws->dimensions()}
    , ws_{ws}
    , fft_{FFT::Make<Rank, FFTRank>(*ws_)}
  {
  }

  InputDims inputDimensions() const { return dims_; }
  OutputDims outputDimensions() const { return dims_; }

  auto forward(Input const &x) const -> Output const &
  {
    auto const start = Log::Now();
    *ws_ = x;
    fft_->forward(*ws_);
    LOG_DEBUG("FFT Forward (Out-of-place) Norm {}->{} Took {}", Norm(x), Norm(*ws_), Log::ToNow(start));
    return *ws_;
  }

  auto adjoint(Output const &x) const -> Input const &
  {
    auto start = Log::Now();
    *ws_ = x;
    fft_->reverse(*ws_);
    LOG_DEBUG(FMT_STRING("FFT Adjoint (Out-of-place) Norm {}->{} Took {}"), Norm(x), Norm(*ws_), Log::ToNow(start));
    return *ws_;
  }

private:
  InputDims dims_;
  std::shared_ptr<Tensor> ws_;
  std::unique_ptr<FFT::FFT<Rank, FFTRank>> fft_;
};
} // namespace rl
