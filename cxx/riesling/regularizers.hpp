#pragma once

#include "args.hpp"

#include "rl/algo/regularizer.hpp"

#include <variant>

namespace rl {

struct RegOpts
{
  RegOpts(args::Subparser &parser);

  args::ValueFlag<std::string> iso;

  args::ValueFlag<float> l1;
  args::ValueFlag<float> l1i;
  args::ValueFlag<float> tv;
  args::ValueFlag<float> tv2;
  args::ValueFlag<float> tgv;
  args::ValueFlag<float> tvt;

  args::ValueFlag<float> llr;
  args::ValueFlag<Index> llrPatch;
  args::ValueFlag<Index> llrWin;
  args::Flag             llrShift;

  args::ValueFlag<float>                                   wavelets;
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> waveDims;
  args::ValueFlag<Index>                                   waveWidth;
};

struct Regularizers_t
{
  std::vector<Regularizer> regs;
  Ops::Op::Ptr             A;
  Ops::Op::Ptr             ext_x;
};

auto Regularizers(RegOpts &regOpts, TOps::TOp<5, 5>::Ptr const &A) -> Regularizers_t;

} // namespace rl
