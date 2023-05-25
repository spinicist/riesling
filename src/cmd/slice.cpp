#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_slice(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to slice");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});

  args::ValueFlag<Index> channelSize(parser, "T", "Number of channels to keep", {"channel-size"}, 0);
  args::ValueFlag<Index> channelStart(parser, "T", "Channel to start split", {"channel-start"}, 0);
  args::ValueFlag<Index> channelStride(parser, "SZ", "Channel stride", {"channel-stride"}, 1);

  args::ValueFlag<Index> readSize(parser, "T", "Number of read samples to keep", {"read-size"}, 0);
  args::ValueFlag<Index> readStart(parser, "T", "Read sample to start split", {"read-start"}, 0);
  args::ValueFlag<Index> readStride(parser, "SZ", "Read sample stride", {"read-stride"}, 1);

  args::ValueFlag<Index> traceStart(parser, "T", "Trace to start split", {"trace-start"}, 0);
  args::ValueFlag<Index> traceSize(parser, "SZ", "Number of traces to keep", {"trace-size"}, 0);
  args::ValueFlag<Index> traceStride(parser, "S", "Trace Stride", {"trace-stride"}, 1);

  args::ValueFlag<Index> volSize(parser, "T", "Number of volumes to keep", {"vol-size"}, 0);
  args::ValueFlag<Index> volStart(parser, "T", "Volume to start split", {"vol-start"}, 0);
  args::ValueFlag<Index> volStride(parser, "SZ", "Volume stride", {"vol-stride"}, 1);

  args::ValueFlag<Index> slabSize(parser, "T", "Number of slabs to keep", {"slab-size"}, 0);
  args::ValueFlag<Index> slabStart(parser, "T", "Slab to start split", {"slab-start"}, 0);
  args::ValueFlag<Index> slabStride(parser, "SZ", "Slab stride", {"slab-stride"}, 1);

  ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get());
  Trajectory traj(reader);
  auto info = traj.info();
  Cx5 ks = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);

  Index const cSt = channelStart.Get();
  Index const rSt = readStart.Get();
  Index const tSt = traceStart.Get();
  Index const sSt = slabStart.Get();
  Index const vSt = volStart.Get();

  Index const cSz = channelSize ? channelSize.Get() : ks.dimension(0) - cSt;
  Index const rSz = readSize ? readSize.Get() : ks.dimension(1) - rSt;
  Index const tSz = traceSize ? traceSize.Get() : ks.dimension(2) - tSt;
  Index const sSz = slabSize ? slabSize.Get() : ks.dimension(3) - sSt;
  Index const vSz = volSize ? volSize.Get() : ks.dimension(4) - vSt;

  if (cSt + cSz > ks.dimension(0)) {
    Log::Fail("Last read point {} exceeded maximum {}", cSt + cSz, ks.dimension(0));
  }
  if (rSt + rSz > ks.dimension(1)) {
    Log::Fail("Last read point {} exceeded maximum {}", rSt + rSz, ks.dimension(1));
  }
  if (tSt + tSz > ks.dimension(2)) {
    Log::Fail("Last trace point {} exceeded maximum {}", tSt + tSz, ks.dimension(2));
  }
  if (sSt + sSz > ks.dimension(3)) {
    Log::Fail("Last slab point {} exceeded maximum {}", sSt + sSz, ks.dimension(3));
  }
  if (vSt + vSz > ks.dimension(4)) {
    Log::Fail("Last volume point {} exceeded maximum {}", vSt + vSz, ks.dimension(4));
  }

  ks = Cx5(ks.slice(Sz5{cSt, rSt, tSt, sSt, vSt}, Sz5{cSz, rSz, tSz, sSz, vSz}));
  traj = Trajectory(info, traj.points().slice(Sz3{0, rSt, tSt}, Sz3{3, rSz, tSz}));

  if (channelStride || readStride || traceStride || slabStride || volStride) {
    ks = Cx5(ks.stride(Sz5{channelStride.Get(), readStride.Get(), traceStride.Get(), slabStride.Get(), volStride.Get()}));
    traj = Trajectory(info, traj.points().stride(Sz3{1, readStride.Get(), traceStride.Get()}));
  }

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), parser.GetCommand().Name()));
  traj.write(writer);
  writer.writeTensor(HD5::Keys::Noncartesian, ks.dimensions(), ks.data());

  return EXIT_SUCCESS;
}
