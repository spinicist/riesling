#include "inputs.hpp"

#include "rl/algo/stats.hpp"
#include "rl/fft.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/pad.hpp"
#include "rl/sense/sense.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_sense_calib(args::Subparser &parser)
{
  CoreArgs<3>                  coreArgs(parser);
  GridArgs<3>                  gridOpts(parser);
  SENSEArgs<3>                 senseArgs(parser);
  args::ValueFlag<std::string> refname(parser, "F", "Reference scan filename", {"ref"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());
  Trajectory  traj(reader, reader.readStruct<Info>(HD5::Keys::Info).voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  auto const basis = LoadBasis(coreArgs.basisFile.Get());

  Cx5 channels = SENSE::LoresChannels(senseArgs.Get(), gridOpts.Get(), traj, noncart, basis.get());
  Cx4 ref;
  if (refname) {
    HD5::Reader refFile(refname.Get());
    Trajectory  refTraj(refFile, refFile.readStruct<Info>(HD5::Keys::Info).voxel_size);
    if (!refTraj.compatible(traj)) { throw Log::Failure(cmd, "Reference data incompatible with multi-channel data"); }
    auto refNoncart = refFile.readTensor<Cx5>();
    if (refNoncart.dimension(0) != 1) { throw Log::Failure(cmd, "Reference data must be single channel"); }
    refTraj.checkDims(FirstN<3>(refNoncart.dimensions()));
    ref = SENSE::LoresChannels(senseArgs.Get(), gridOpts.Get(), refTraj, refNoncart, basis.get()).chip<4>(0).abs().cast<Cx>();
  } else {
    ref = DimDot<3>(channels, channels).sqrt();
  }
  Cx5 const kernels = SENSE::EstimateKernels<3>(channels, ref, senseArgs.kWidth.Get(), gridOpts.osamp.Get(), senseArgs.l.Get(),
                                                senseArgs.λ.Get());
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeTensor(HD5::Keys::Data, kernels.dimensions(), kernels.data(), HD5::Dims::SENSE);
  writer.writeTensor("channels", channels.dimensions(), channels.data(), HD5::Dims::SENSE);
  writer.writeTensor("ref", ref.dimensions(), ref.data(), {"i", "j", "k", "b"});
  Log::Print(cmd, "Finished");
}

void main_sense_calib2(args::Subparser &parser)
{
  CoreArgs<2>                  coreArgs(parser);
  GridArgs<2>                  gridOpts(parser);
  SENSEArgs<2>                 senseArgs(parser);
  args::ValueFlag<std::string> refname(parser, "F", "Reference scan filename", {"ref"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const     cmd = parser.GetCommand().Name();
  HD5::Reader    reader(coreArgs.iname.Get());
  TrajectoryN<2> traj(reader, reader.readStruct<Info>(HD5::Keys::Info).voxel_size.head<2>());
  auto           noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  auto const basis = LoadBasis(coreArgs.basisFile.Get());

  Cx5 channels = SENSE::LoresChannels<2>(senseArgs.Get(), gridOpts.Get(), traj, noncart, basis.get());
  Cx4 ref;
  if (refname) {
    HD5::Reader    refFile(refname.Get());
    TrajectoryN<2> refTraj(refFile, refFile.readStruct<Info>(HD5::Keys::Info).voxel_size.head<2>());
    if (!refTraj.compatible(traj)) { throw Log::Failure(cmd, "Reference data incompatible with multi-channel data"); }
    auto refNoncart = refFile.readTensor<Cx5>();
    if (refNoncart.dimension(0) != 1) { throw Log::Failure(cmd, "Reference data must be single channel"); }
    refTraj.checkDims(FirstN<3>(refNoncart.dimensions()));
    ref =
      SENSE::LoresChannels<2>(senseArgs.Get(), gridOpts.Get(), refTraj, refNoncart, basis.get()).chip<4>(0).abs().cast<Cx>();
  } else {
    ref = DimDot<3>(channels, channels).sqrt();
  }
  Cx5 const kernels = SENSE::EstimateKernels<2>(channels, ref, senseArgs.kWidth.Get(), gridOpts.osamp.Get(), senseArgs.l.Get(),
                                                senseArgs.λ.Get());
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeTensor(HD5::Keys::Data, kernels.dimensions(), kernels.data(), HD5::Dims::SENSE);
  writer.writeTensor("channels", channels.dimensions(), channels.data(), HD5::Dims::SENSE);
  writer.writeTensor("ref", ref.dimensions(), ref.data(), HD5::DNames<4>{"i", "j", "k", "b"});
  Log::Print(cmd, "Finished");
}
