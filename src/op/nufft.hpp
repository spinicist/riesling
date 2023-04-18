#pragma once

#include "op/tensorop.hpp"

#include "apodize.hpp"
#include "fft/fft.hpp"
#include "sdc.hpp"
#include "make_grid.hpp"
#include "pad.hpp"

namespace rl {

template <size_t NDim>
struct NUFFTOp final : TensorOperator<Cx, NDim + 2, 3>
{
  OP_INHERIT(Cx, NDim + 2, 3)

  NUFFTOp(
    std::shared_ptr<GridBase<Cx, NDim>> gridder,
    Sz<NDim> const matrix,
    std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr,
    bool toeplitz = false);

  auto forward(InTensor const &x) const -> OutTensor;
  auto adjoint(OutTensor const &y) const -> InTensor;
  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;

  // auto adjfwd(InCMap x) const -> InputMap;
  // auto fft() const -> FFTOp<NDim + 2, NDim> const &;

  std::shared_ptr<GridBase<Cx, NDim>> gridder;
  InTensor mutable workspace;
  std::shared_ptr<FFT::FFT<NDim + 2, NDim>> fft;

  PadOp<Cx, NDim + 2, NDim> pad;
  ApodizeOp<NDim> apo;
  std::shared_ptr<TensorOperator<Cx, 3>> sdc;
private:
  using Transfer = Eigen::Tensor<Cx, NDim + 2>;
  Transfer tf_;
};

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_nufft(
  Trajectory const &traj,
  std::string const &ktype,
  float const osamp,
  Index const nC,
  Sz3 const matrix,
  std::optional<Re2> basis = std::nullopt,
  std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr,
  bool const toeplitz = false);

} // namespace rl
