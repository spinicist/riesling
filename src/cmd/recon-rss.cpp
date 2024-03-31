#include "types.hpp"

#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "op/rank.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_recon_rss(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  GridOpts                     gridOpts(parser);
  SDC::Opts                    sdcOpts(parser, "pipe");
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});
  args::ValueFlag<float>       t0(parser, "T0", "Time of first sample for off-resonance correction", {"t0"});
  args::ValueFlag<float>       tSamp(parser, "TS", "Sample time for off-resonance correction", {"tsamp"});

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  Index const nC = reader.dimensions(HD5::Keys::Noncartesian)[0];
  auto const  sdc = SDC::Choose(sdcOpts, nC, traj, gridOpts.ktype.Get(), gridOpts.osamp.Get());

  std::shared_ptr<TensorOperator<Cx, 5, 4>> A = nullptr;
  if (coreOpts.ndft) {
    auto ndft = std::make_shared<NDFTOp<3>>(traj.points(), nC, traj.matrixForFOV(coreOpts.fov.Get()), basis, sdc);
    if (reader.exists("b0") && t0 && tSamp) { ndft->addOffResonance(reader.readTensor<Re3>("b0"), t0.Get(), tSamp.Get()); }
    A = std::make_shared<IncreaseOutputRank<NDFTOp<3>>>(ndft);
  } else {
    A = make_nufft(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, traj.matrixForFOV(coreOpts.fov.Get()), basis, sdc);
  }
  Sz4 sz = LastN<4>(A->ishape);

  Cx5         allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  traj.checkDims(FirstN<3>(allData.dimensions()));
  Index const volumes = allData.dimension(4);
  Cx5         out(AddBack(sz, volumes));
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto const channels = A->adjoint(CChipMap(allData, iv));
    out.chip<4>(iv) = ConjugateSum(channels, channels).sqrt();
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved());
  Log::Print("Finished {}", parser.GetCommand().Name());
  return EXIT_SUCCESS;
}
