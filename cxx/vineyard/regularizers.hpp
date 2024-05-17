#pragma once

#include "op/ops.hpp"
#include "parse_args.hpp"
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

  args::ValueFlag<float>            wavelets;
  args::ValueFlag<Sz4, SzReader<4>> waveDims;
  args::ValueFlag<Index>            waveWidth;
};

struct Regularizer
{
  Ops::Op<Cx>::Ptr     T;
  Proxs::Prox<Cx>::Ptr P;
};

struct Regularizers
{
  using SizeN = std::variant<Sz3, Sz4, Sz5>;
  std::vector<Regularizer>     regs;
  std::shared_ptr<Ops::Op<Cx>> ext_x;
  std::vector<SizeN>           sizes;

  Regularizers(RegOpts &regOpts, Sz4 const shape, std::shared_ptr<Ops::Op<Cx>> &A);
};

} // namespace rl
