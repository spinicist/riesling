#pragma once

#include "operator.hpp"

#include "apodize.hpp"
#include "fft.hpp"
#include "sdc.hpp"
#include "make_grid.hpp"
#include "pad.hpp"

namespace rl {

template <size_t NDim>
struct NUFFTOp final : Operator<Cx, NDim + 2, 3>
{
  OP_INHERIT(Cx, NDim + 2, 3)

  NUFFTOp(
    std::unique_ptr<GridBase<Cx, NDim>> gridder,
    Sz<NDim> const matrix,
    std::optional<SDC::Functor> sdc = std::nullopt,
    bool toeplitz = false);

  OP_DECLARE()

  auto adjfwd(InputMap x) const -> InputMap;
  auto fft() const -> FFTOp<NDim + 2, NDim> const &;

private:
  std::unique_ptr<GridBase<Cx, NDim>> gridder_;
  FFTOp<NDim + 2, NDim> fft_;
  PadOp<Cx, NDim + 2, NDim> pad_;
  ApodizeOp<Cx, NDim> apo_;
  std::optional<SDC::Functor> sdc_;
  using Transfer = Eigen::Tensor<Cx, NDim + 2>;
  Transfer tf_;
};

std::unique_ptr<Operator<Cx, 5, 4>> make_nufft(
  Trajectory const &traj,
  std::string const &ktype,
  float const osamp,
  Index const nC,
  Sz3 const matrix,
  std::optional<SDC::Functor> sdc = std::nullopt,
  std::optional<Re2> basis = std::nullopt,
  bool const toeplitz = false);

} // namespace rl
