#pragma once

#include "op/tensorop.hpp"

#include "apodize.hpp"
#include "fft/fft.hpp"
#include "grid.hpp"
#include "pad.hpp"
#include "sdc.hpp"

namespace rl {

template <size_t NDim>
struct NUFFTOp final : TensorOperator<Cx, NDim + 2, 3>
{
  OP_INHERIT(Cx, NDim + 2, 3)

  NUFFTOp(std::shared_ptr<Grid<Cx, NDim>> gridder, Sz<NDim> const matrix, std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr);

  OP_DECLARE()

  // auto adjfwd(InCMap x) const -> InputMap;
  // auto fft() const -> FFTOp<NDim + 2, NDim> const &;

  std::shared_ptr<Grid<Cx, NDim>> gridder;
  InTensor mutable workspace;
  std::shared_ptr<FFT::FFT<NDim + 2, NDim>> fft;

  PadOp<Cx, NDim + 2, NDim>              pad;
  ApodizeOp<Cx, NDim>                    apo;
  std::shared_ptr<TensorOperator<Cx, 3>> sdc;

private:
  using Transfer = Eigen::Tensor<Cx, NDim + 2>;
  Transfer tf_;
};

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_nufft(Trajectory const                      &traj,
                                                     std::string const                     &ktype,
                                                     float const                            osamp,
                                                     Index const                            nC,
                                                     Sz3 const                              matrix,
                                                     Re2 const                             &basis = IdBasis(),
                                                     std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr,
                                                     Index const                            bSz = 32,
                                                     Index const                            sSz = 16384);

} // namespace rl
