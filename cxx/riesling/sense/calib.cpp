#include "types.hpp"

#include "algo/stats.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sense/sense.hpp"
#include "fft.hpp"

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

  Cx5 channels = SENSE::LoresKernels(senseOpts, gridOpts, traj, noncart, basis);
  Cx4 ref;
  if (refname) {
    HD5::Reader refFile(refname.Get());
    Trajectory  refTraj(refFile, refFile.readInfo().voxel_size);
    if (!refTraj.compatible(traj)) { Log::Fail("Reference data incompatible with multi-channel data"); }
    auto refNoncart = refFile.readTensor<Cx5>();
    if (refNoncart.dimension(0) != 1) { Log::Fail("Reference data must be single channel"); }
    refTraj.checkDims(FirstN<3>(refNoncart.dimensions()));
    ref = SENSE::LoresKernels(senseOpts, gridOpts, refTraj, refNoncart, basis).chip<0>(0);

    // Normalize energy
    ref = ref * ref.constant(Norm(channels) / Norm(ref));
  } else {
    auto const phases = FFT::PhaseShift(LastN<3>(channels.dimensions()));
    Cx5 temp = channels;
    FFT::Adjoint<5, 3>(temp, Sz3{2, 3, 4}, phases);
    ref = ConjugateSum(temp, temp);
    FFT::Forward<4, 3>(ref, Sz3{1, 2, 3}, phases);
  }
  
  SENSE::Nonsense(channels, ref);

  if (frame) {
    auto shape = channels.dimensions();
    if (frame.Get() < 0 || frame.Get() >= shape[1]) {
      Log::Fail("Requested frame {} is outside valid range 0-{}", frame.Get(), shape[1]);
    }
    shape[1] = 1;
    channels = Cx5(channels.slice(Sz5{0, frame.Get(), 0, 0, 0}, shape));
  }
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor(HD5::Keys::Data, channels.dimensions(), channels.data(), HD5::Dims::SENSE);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
