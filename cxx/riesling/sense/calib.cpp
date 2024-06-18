#include "types.hpp"

#include "algo/stats.hpp"
#include "fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "pad.hpp"
#include "parse_args.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_sense_calib(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  GridOpts                     gridOpts(parser);
  SENSE::Opts                  senseOpts(parser);
  args::ValueFlag<std::string> refname(parser, "F", "Reference scan filename", {"ref"});
  args::ValueFlag<Index>       frame(parser, "F", "SENSE calibration frame (all)", {"frame"}, -1);
  args::Flag                   nonsense(parser, "N", "NonSENSE", {'n', "nonsense"});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  auto const basis = ReadBasis(coreOpts.basisFile.Get());

  Cx5 channels = SENSE::LoresChannels(senseOpts, gridOpts, traj, noncart, basis);
  Cx4 ref;
  if (refname) {
    HD5::Reader refFile(refname.Get());
    Trajectory  refTraj(refFile, refFile.readInfo().voxel_size);
    if (!refTraj.compatible(traj)) { Log::Fail("Reference data incompatible with multi-channel data"); }
    auto refNoncart = refFile.readTensor<Cx5>();
    if (refNoncart.dimension(0) != 1) { Log::Fail("Reference data must be single channel"); }
    refTraj.checkDims(FirstN<3>(refNoncart.dimensions()));
    ref = SENSE::LoresChannels(senseOpts, gridOpts, refTraj, refNoncart, basis).chip<0>(0);

    // Normalize energy
    ref = ref * ref.constant(Norm(channels) / Norm(ref));
  } else {
    ref = ConjugateSum(channels, channels);
  }

  auto maps = SENSE::Nonsense(channels, ref, senseOpts.kWidth.Get() * gridOpts.osamp.Get());
  if (frame) {
    auto shape = maps.dimensions();
    if (frame.Get() < 0 || frame.Get() >= shape[1]) {
      Log::Fail("Requested frame {} is outside valid range 0-{}", frame.Get(), shape[1]);
    }
    shape[1] = 1;
    maps = Cx5(maps.slice(Sz5{0, frame.Get(), 0, 0, 0}, shape));
  }

  // Pad out to full SENSE map size
  // Sz5 const ksize = kernels.dimensions();
  // Sz3 const mat = traj.matrixForFOV(senseOpts.fov.Get());
  // Sz5 const mapsize = AddFront(mat, ksize[0], ksize[1]);
  // Cx5       maps = Pad(kernels, mapsize);
  // FFT::Adjoint<5, 3>(maps, Sz3{2, 3, 4}, FFT::PhaseShift(mat));
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor(HD5::Keys::Data, maps.dimensions(), maps.data(), HD5::Dims::SENSE);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
