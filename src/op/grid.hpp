#pragma once

#include "basis/basis.hpp"
#include "kernel/kernel.hpp"
#include "mapping.hpp"
#include "parse_args.hpp"
#include "tensorop.hpp"
#include "threads.hpp"

#include <mutex>

namespace rl {

struct GridOpts
{
  GridOpts(args::Subparser &parser);
  args::ValueFlag<std::string> ktype;
  args::ValueFlag<float>       osamp;
  args::ValueFlag<Index>       bucketSize, splitSize;
};

template <typename Scalar_, int NDim>
struct Grid final : TensorOperator<Scalar_, NDim + 2, 3>
{
  OP_INHERIT(Scalar_, NDim + 2, 3)
  using Parent::adjoint;
  using Parent::forward;
  using T1 = Eigen::Tensor<Scalar, 1>;
  std::shared_ptr<Kernel<Scalar, NDim>> kernel;
  Mapping<NDim>                         mapping;
  Basis<Scalar>                         basis;

  static auto Make(Trajectory const    &t,
                   std::string const    kt,
                   float const          os,
                   Index const          nC = 1,
                   Basis<Scalar> const &b = IdBasis<Scalar>(),
                   Index const          bSz = 32,
                   Index const          sSz = 16384) -> std::shared_ptr<Grid<Scalar, NDim>>;
  Grid(std::shared_ptr<Kernel<Scalar, NDim>> const &k,
       Mapping<NDim> const                          m,
       Index const                                  nC,
       Basis<Scalar> const                         &b = IdBasis<Scalar>());
  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;
};

} // namespace rl
