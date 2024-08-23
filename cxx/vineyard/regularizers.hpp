#pragma once

#include "args.hpp"
#include "op/top.hpp"
#include "prox/prox.hpp"

#include <variant>

namespace rl {

struct RegOpts
{
  RegOpts(args::Subparser &parser);

  args::ValueFlag<float> l1;
  args::ValueFlag<float> nmrent;

  args::ValueFlag<float> tv;
  args::ValueFlag<float> tvt;

  args::ValueFlag<float> tgv;
  args::ValueFlag<float> tgvl2;

  args::ValueFlag<float> llr;
  args::ValueFlag<Index> llrPatch;
  args::ValueFlag<Index> llrWin;
  args::Flag             llrShift;

  args::ValueFlag<float>                                   wavelets;
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> waveDims;
  args::ValueFlag<Index>                                   waveWidth;
};

struct Regularizer
{
  using SizeN = std::variant<Sz4, Sz5, Sz6>;
  Ops::Op<Cx>::Ptr     T;
  Proxs::Prox<Cx>::Ptr P;
  SizeN                size;
};

struct Regularizers_t
{
  std::vector<Regularizer> regs;
  Ops::Op<Cx>::Ptr         A;
  Ops::Op<Cx>::Ptr         ext_x;
};

auto Regularizers(RegOpts &regOpts, TOps::TOp<Cx, 5, 5>::Ptr const &A) -> Regularizers_t;

} // namespace rl
