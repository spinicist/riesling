#include "types.hpp"

#include "algo/lsmr.hpp"
#include "cropper.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "op/rank.hpp"
#include "parse_args.hpp"
#include "precon.hpp"
#include "tensorOps.hpp"

using namespace rl;

void main_recon_rss(args::Subparser &parser)
{
  CoreOpts               coreOpts(parser);
  GridOpts               gridOpts(parser);
  PreconOpts             preOpts(parser);
  LsqOpts                lsqOpts(parser);
  args::ValueFlag<float> t0(parser, "T0", "Time of first sample for off-resonance correction", {"t0"});
  args::ValueFlag<float> tSamp(parser, "TS", "Sample time for off-resonance correction", {"tsamp"});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  Index const nC = reader.dimensions(HD5::Keys::Data)[0];

  std::shared_ptr<TensorOperator<Cx, 5, 4>> A = nullptr;
  if (coreOpts.ndft) {
    auto ndft = std::make_shared<NDFTOp<3>>(traj.points(), nC, traj.matrixForFOV(coreOpts.fov.Get()), basis);
    if (reader.exists("b0") && t0 && tSamp) { ndft->addOffResonance(reader.readTensor<Re3>("b0"), t0.Get(), tSamp.Get()); }
    A = std::make_shared<IncreaseOutputRank<NDFTOp<3>>>(ndft);
  } else {
    A = make_nufft(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, traj.matrixForFOV(coreOpts.fov.Get()), basis);
  }

  auto const M = make_kspace_pre(traj, nC, basis, preOpts.type.Get(), preOpts.bias.Get());
  LSMR const lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

  Cx5 noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nVol = noncart.dimension(4);
  Cx5         out(AddBack(LastN<4>(A->ishape), nVol));
  for (Index iv = 0; iv < nVol; iv++) {
    auto const channels = lsmr.run(&noncart(0, 0, 0, 0, iv));
    auto const channelsT = Tensorfy(channels, A->ishape);
    out.chip<4>(iv) = (channelsT * channelsT.conjugate()).sum(Sz1{0}).sqrt();
  }
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  Log::Print("Finished {}", parser.GetCommand().Name());
}
