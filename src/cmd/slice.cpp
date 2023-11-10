#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_slice(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to slice");
  args::ValueFlag<std::string>  oname(parser, "OUTPUT", "Override output name", {'o', "out"});

  args::ValueFlag<Index> channelSize(parser, "T", "Number of channels to keep", {"channel-size"}, 0);
  args::ValueFlag<Index> channelStart(parser, "T", "Channel to start split", {"channel-start"}, 0);
  args::ValueFlag<Index> channelStride(parser, "SZ", "Channel stride", {"channel-stride"}, 1);

  args::ValueFlag<Index> readSize(parser, "T", "Number of read samples to keep", {"read-size"}, 0);
  args::ValueFlag<Index> readStart(parser, "T", "Read sample to start split", {"read-start"}, 0);
  args::ValueFlag<Index> readStride(parser, "SZ", "Read sample stride", {"read-stride"}, 1);

  args::ValueFlag<Index> traceStart(parser, "T", "Trace to start split", {"trace-start"}, 0);
  args::ValueFlag<Index> traceSize(parser, "SZ", "Number of traces to keep", {"trace-size"}, 0);
  args::ValueFlag<Index> traceStride(parser, "S", "Trace Stride", {"trace-stride"}, 1);
  args::ValueFlag<Index> traceSegment(parser, "S", "Trace segment size", {"trace-segment"});

  args::ValueFlag<Index> volSize(parser, "T", "Number of volumes to keep", {"vol-size"}, 0);
  args::ValueFlag<Index> volStart(parser, "T", "Volume to start split", {"vol-start"}, 0);
  args::ValueFlag<Index> volStride(parser, "SZ", "Volume stride", {"vol-stride"}, 1);

  args::ValueFlag<Index> slabSize(parser, "T", "Number of slabs to keep", {"slab-size"}, 0);
  args::ValueFlag<Index> slabStart(parser, "T", "Slab to start split", {"slab-start"}, 0);
  args::ValueFlag<Index> slabStride(parser, "SZ", "Slab stride", {"slab-stride"}, 1);

  ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto        info = traj.info();
  Cx5         ks = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);

  Index const cSt = Wrap(channelStart.Get(), ks.dimension(0));
  Index const rSt = Wrap(readStart.Get(), ks.dimension(1));
  Index const tSt = Wrap(traceStart.Get(), ks.dimension(2));
  Index const sSt = Wrap(slabStart.Get(), ks.dimension(3));
  Index const vSt = Wrap(volStart.Get(), ks.dimension(4));

  Index const cSz = channelSize ? channelSize.Get() : ks.dimension(0) - cSt;
  Index const rSz = readSize ? readSize.Get() : ks.dimension(1) - rSt;
  Index const tSz = traceSize ? traceSize.Get() : ks.dimension(2) - tSt;
  Index const sSz = slabSize ? slabSize.Get() : ks.dimension(3) - sSt;
  Index const vSz = volSize ? volSize.Get() : ks.dimension(4) - vSt;

  if (cSt + cSz > ks.dimension(0)) { Log::Fail("Last read point {} exceeded maximum {}", cSt + cSz, ks.dimension(0)); }
  if (rSt + rSz > ks.dimension(1)) { Log::Fail("Last read point {} exceeded maximum {}", rSt + rSz, ks.dimension(1)); }
  if (traceSegment) {
    if (tSt + tSz > traceSegment.Get()) {
      Log::Fail("Last trace point {} exceeded segment size {}", tSt + tSz, traceSegment.Get());
    }
  } else {
    if (tSt + tSz > ks.dimension(2)) { Log::Fail("Last trace point {} exceeded maximum {}", tSt + tSz, ks.dimension(2)); }
  }
  if (sSt + sSz > ks.dimension(3)) { Log::Fail("Last slab point {} exceeded maximum {}", sSt + sSz, ks.dimension(3)); }
  if (vSt + vSz > ks.dimension(4)) { Log::Fail("Last volume point {} exceeded maximum {}", vSt + vSz, ks.dimension(4)); }

  Log::Print("Selected slice {}:{}, {}:{}, {}:{}, {}:{}, {}:{}", cSt, cSt + cSz - 1, rSt, rSt + rSz - 1, tSt, tSt + tSz - 1,
             sSt, sSt + sSz - 1, vSt, vSt + vSz - 1);

  if (traceSegment) {
    Index const segSz = traceSegment.Get();
    Index const nSeg = ks.dimension(2) / segSz; // Will lose spare traces
    Sz5 const   shape5{cSz, rSz, tSz * nSeg, sSz, vSz};
    Sz6 const   shape6{ks.dimension(0), ks.dimension(1), segSz, nSeg, ks.dimension(3), ks.dimension(4)};
    ks = Cx5(ks.reshape(shape6).slice(Sz6{cSt, rSt, tSt, 0, sSt, vSt}, Sz6{cSz, rSz, tSz, nSeg, sSz, vSz}).reshape(shape5));
    Sz3 const shape3{3, rSz, tSz * nSeg};
    Sz4 const shape4{3, traj.nSamples(), segSz, nSeg};
    traj = Trajectory(info, traj.points().reshape(shape4).slice(Sz4{0, rSt, tSt, 0}, Sz4{3, rSz, tSz, nSeg}).reshape(shape3));
  } else {
    ks = Cx5(ks.slice(Sz5{cSt, rSt, tSt, sSt, vSt}, Sz5{cSz, rSz, tSz, sSz, vSz}));
    traj = Trajectory(info, traj.points().slice(Sz3{0, rSt, tSt}, Sz3{3, rSz, tSz}));
  }

  if (channelStride || readStride || traceStride || slabStride || volStride) {
    ks = Cx5(ks.stride(Sz5{channelStride.Get(), readStride.Get(), traceStride.Get(), slabStride.Get(), volStride.Get()}));
    traj = Trajectory(info, traj.points().stride(Sz3{1, readStride.Get(), traceStride.Get()}));
  }

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), parser.GetCommand().Name()));
  writer.writeInfo(traj.info());
  writer.writeTensor(HD5::Keys::Trajectory, traj.points().dimensions(), traj.points().data());
  writer.writeTensor(HD5::Keys::Noncartesian, ks.dimensions(), ks.data());

  return EXIT_SUCCESS;
}
