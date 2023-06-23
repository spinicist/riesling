#pragma once

#include "parse_args.hpp"
#include "op/ops.hpp"
#include "prox/prox.hpp"

namespace rl {

struct RegOpts
{
  RegOpts(args::Subparser &parser);

  args::ValueFlag<float> tv;
  args::ValueFlag<float> tvt;
  args::ValueFlag<float> tgv;
  args::ValueFlag<float> l1;
  args::ValueFlag<float> nmrent;

  args::ValueFlag<float> llr;
  args::ValueFlag<Index> llrPatch;
  args::ValueFlag<Index> llrWin;

  args::ValueFlag<Index> wavelets;
  args::ValueFlag<Index> waveLevels;
  args::ValueFlag<Index> waveWidth;
};

struct Regularizers {
    Regularizers(RegOpts &regOpts, Sz4 const shape, std::shared_ptr<Ops::Op<Cx>> &A);

  std::vector<std::shared_ptr<Ops::Op<Cx>>> ops;
  std::vector<std::shared_ptr<Proxs::Prox<Cx>>> prox;
  std::shared_ptr<Ops::Op<Cx>> ext_x;

  auto σ(std::vector<float> σin) const -> std::vector<float>;
};


}
