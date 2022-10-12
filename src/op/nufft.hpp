#pragma once

#include "operator.hpp"

#include "apodize.hpp"
#include "fft.hpp"
#include "make_grid.hpp"
#include "pad.hpp"
#include "sdc.hpp"

namespace rl {

template <size_t NDim>
struct NUFFTOp final : Operator<NDim + 2, 3>
{
  using Input = typename Operator<NDim + 2, 3>::Input;
  using Output = typename Operator<NDim + 2, 3>::Output;
  using InputDims = typename Operator<NDim + 2, 3>::InputDims;
  using OutputDims = typename Operator<NDim + 2, 3>::OutputDims;

  NUFFTOp(
    Trajectory const &traj,
    std::string const &ktype,
    float const osamp,
    Index const nC,
    Sz<NDim> const matrix,
    Operator<3, 3> *sdc = nullptr,
    std::optional<Re2> basis = std::nullopt,
    bool toeplitz = false);

  auto inputDimensions() const -> InputDims;
  auto outputDimensions() const -> OutputDims;
  auto forward(Input const &x) const -> Output const &;
  auto adjoint(Output const &x) const -> Input const &;
  auto adjfwd(Input const &x) const -> Input;
  auto fft() const -> FFTOp<NDim + 2, NDim> const &;

private:
  std::unique_ptr<GridBase<Cx, NDim>> gridder_;
  FFTOp<NDim + 2, NDim> fft_;
  PadOp<NDim + 2, NDim> pad_;
  ApodizeOp<Cx, NDim> apo_;
  Operator<3, 3> *sdc_;
  using Transfer = Eigen::Tensor<Cx, NDim + 2>;
  Transfer tf_;

};

std::unique_ptr<Operator<5, 4>> make_nufft(
  Trajectory const &traj,
  std::string const &ktype,
  float const osamp,
  Index const nC,
  Sz3 const matrix,
  Operator<3, 3> *sdc = nullptr,
  std::optional<Re2> basis = std::nullopt,
  bool const toeplitz = false);

} // namespace rl
