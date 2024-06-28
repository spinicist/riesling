#include "types.hpp"

#include "algo/stats.hpp"
#include "fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/fft.hpp"
#include "op/pad.hpp"
#include "parse_args.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_sense_calib(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  GridOpts                     gridOpts(parser);
  SENSE::Opts                  senseOpts(parser);
  args::ValueFlag<std::string> refname(parser, "F", "Reference scan filename", {"ref"});
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
    channels = channels * channels.constant(std::sqrt(Product(ref.dimensions())) / Norm(channels));
    ref = ref * ref.constant(std::sqrt(Product(ref.dimensions())) / Norm(ref));
  } else {
    ref = ConjugateSum(channels, channels).sqrt();
  }
  HD5::Writer writer(coreOpts.oname.Get());
  auto              kernels = SENSE::Nonsense(channels, ref, senseOpts.kWidth.Get(), senseOpts.λ.Get());
  auto const        kshape = kernels.dimensions();
  auto const        fshape = AddFront(traj.matrix(gridOpts.osamp.Get()), kshape[0], kshape[1]);
  auto const        cshape = AddFront(traj.matrixForFOV(senseOpts.fov.Get()), kshape[0], kshape[1]);
  TOps::Pad<Cx, 5>  P(kshape, fshape);
  TOps::FFT<5, 3>   F(fshape, false);
  TOps::Crop<Cx, 5> C(fshape, cshape);
  Cx5 const         maps =
    C.forward(F.adjoint(P.forward(kernels))) * Cx(std::sqrt(Product(LastN<3>(fshape)) / (float)Product(LastN<3>(kshape))));
  writer.writeTensor(HD5::Keys::Data, maps.dimensions(), maps.data(), HD5::Dims::SENSE);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
