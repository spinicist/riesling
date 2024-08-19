#include "types.hpp"

#include "algo/stats.hpp"
#include "fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/fft.hpp"
#include "op/pad.hpp"
#include "inputs.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_sense_calib(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  GridOpts                     gridOpts(parser);
  SENSE::Opts                  senseOpts(parser);
  args::ValueFlag<std::string> refname(parser, "F", "Reference scan filename", {"ref"});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  auto const basis = LoadBasis(coreOpts.basisFile.Get());

  Cx5 channels = SENSE::LoresChannels(senseOpts, gridOpts, traj, noncart, basis.get());
  Cx4 ref;
  if (refname) {
    HD5::Reader refFile(refname.Get());
    Trajectory  refTraj(refFile, refFile.readInfo().voxel_size);
    if (!refTraj.compatible(traj)) { Log::Fail("Reference data incompatible with multi-channel data"); }
    auto refNoncart = refFile.readTensor<Cx5>();
    if (refNoncart.dimension(0) != 1) { Log::Fail("Reference data must be single channel"); }
    refTraj.checkDims(FirstN<3>(refNoncart.dimensions()));
    ref = SENSE::LoresChannels(senseOpts, gridOpts, refTraj, refNoncart, basis.get()).chip<0>(0);
    // Normalize energy
    channels = channels * channels.constant(std::sqrt(Product(ref.dimensions())) / Norm(channels));
    ref = ref * ref.constant(std::sqrt(Product(ref.dimensions())) / Norm(ref));
  } else {
    ref = DimDot<1>(channels, channels).sqrt();
  }
  Cx5 const   kernels = SENSE::EstimateKernels(channels, ref, senseOpts.kWidth.Get(), senseOpts.λ.Get());
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor(HD5::Keys::Data, kernels.dimensions(), kernels.data(), HD5::Dims::SENSE);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
