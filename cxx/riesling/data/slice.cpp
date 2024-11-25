#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_slice(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Index> channelStart(parser, "T", "Channel to start slice", {"channel-start"}, 0);
  args::ValueFlag<Index> channelSize(parser, "T", "Number of channels to keep", {"channel-size"}, 0);
  args::ValueFlag<Index> channelStride(parser, "SZ", "Channel stride", {"channel-stride"}, 1);

  args::ValueFlag<Index> sampleStart(parser, "T", "Read sample to start slice", {"sample-start"}, 0);
  args::ValueFlag<Index> sampleSize(parser, "T", "Number of read samples to keep", {"sample-size"}, 0);
  args::ValueFlag<Index> sampleStride(parser, "SZ", "Read sample stride", {"sample-stride"}, 1);

  args::ValueFlag<Index> traceStart(parser, "T", "Trace to start slice", {"trace-start"}, 0);
  args::ValueFlag<Index> traceSize(parser, "SZ", "Number of traces to keep", {"trace-size"}, 0);
  args::ValueFlag<Index> traceStride(parser, "S", "Trace Stride", {"trace-stride"}, 1);

  args::ValueFlag<Index> tracesPerSeg(parser, "S", "Trace segment size", {"traces-per-seg"}, 0);
  args::ValueFlag<Index> traceSegStart(parser, "S", "Trace segment start", {"seg-start"}, 0);
  args::ValueFlag<Index> traceSegments(parser, "S", "Trace segments", {"seg-size"});

  args::ValueFlag<Index> volStart(parser, "T", "Volume to start slice", {"vol-start"}, 0);
  args::ValueFlag<Index> volSize(parser, "T", "Number of volumes to keep", {"vol-size"}, 0);
  args::ValueFlag<Index> volStride(parser, "SZ", "Volume stride", {"vol-stride"}, 1);

  args::ValueFlag<Index> slabStart(parser, "T", "Slab to start slice", {"slab-start"}, 0);
  args::ValueFlag<Index> slabSize(parser, "T", "Number of slabs to keep", {"slab-size"}, 0);
  args::ValueFlag<Index> slabStride(parser, "SZ", "Slab stride", {"slab-stride"}, 1);

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  auto const  info = reader.readInfo();

  if (reader.order() != 5) { throw Log::Failure(cmd, "Dataset does not appear to be non-cartesian with 5 dimensions"); }
  auto const shape = reader.dimensions();

  Index const cSt = Wrap(channelStart.Get(), shape[0]);
  Index const rSt = Wrap(sampleStart.Get(), shape[1]);
  Index const tSt = Wrap(traceStart.Get(), shape[2]);
  Index const sSt = Wrap(slabStart.Get(), shape[3]);
  Index const vSt = Wrap(volStart.Get(), shape[4]);

  Index const cSz = channelSize ? channelSize.Get() : shape[0] - cSt;
  Index const rSz = sampleSize ? sampleSize.Get() : shape[1] - rSt;
  Index const tSz = traceSize ? traceSize.Get() : (tracesPerSeg ? tracesPerSeg.Get() - tSt : shape[2] - tSt);
  Index const sSz = slabSize ? slabSize.Get() : shape[3] - sSt;
  Index const vSz = volSize ? volSize.Get() : shape[4] - vSt;

  if (cSt + cSz > shape[0]) { throw Log::Failure(cmd, "Last channel {} exceeded maximum {}", cSt + cSz, shape[0]); }
  if (rSt + rSz > shape[1]) { throw Log::Failure(cmd, "Last sample {} exceeded maximum {}", rSt + rSz, shape[1]); }
  if (tracesPerSeg) {
    if (tSt + tSz > tracesPerSeg.Get()) {
      throw Log::Failure(cmd, "Last trace {} exceeded segment size {}", tSt + tSz, tracesPerSeg.Get());
    }
  } else {
    if (tSt + tSz > shape[2]) { throw Log::Failure(cmd, "Last trace {} exceeded maximum {}", tSt + tSz, shape[2]); }
  }
  if (sSt + sSz > shape[3]) { throw Log::Failure(cmd, "Last slab {} exceeded maximum {}", sSt + sSz, shape[3]); }
  if (vSt + vSz > shape[4]) { throw Log::Failure(cmd, "Last volume {} exceeded maximum {}", vSt + vSz, shape[4]); }

  if (cSz < 1) { throw Log::Failure(cmd, "Channel size was less than 1"); }
  if (rSz < 1) { throw Log::Failure(cmd, "Sample size was less than 1"); }
  if (tSz < 1) { throw Log::Failure(cmd, "Trace size was less than 1"); }
  if (sSz < 1) { throw Log::Failure(cmd, "Slab size was less than 1"); }
  if (vSz < 1) { throw Log::Failure(cmd, "Volume size was less than 1"); }

  Log::Print(cmd, "Selected slice {}:{}, {}:{}, {}:{}, {}:{}, {}:{}", cSt, cSt + cSz - 1, rSt, rSt + rSz - 1, tSt,
             tSt + tSz - 1, sSt, sSt + sSz - 1, vSt, vSt + vSz - 1);

  Cx5        ks = reader.readTensor<Cx5>();
  Trajectory traj(reader, info.voxel_size);

  if (tracesPerSeg) {
    Index const tps = tracesPerSeg.Get();
    if (tSt + tSz > tps) { throw Log::Failure(cmd, "Selected traces {}-{} extend past segment {}", tSt, tSz, tps); }
    Index const nSeg = shape[2] / tps;
    if (nSeg * tps != shape[2]) {
      throw Log::Failure(cmd, "Traces per seg {} does not cleanly divide traces {}", tps, shape[2]);
    }
    Index const segSt = Wrap(traceSegStart.Get(), nSeg);
    Index const segSz = traceSegments ? std::clamp(traceSegments.Get(), 1L, nSeg) : nSeg - segSt;
    Log::Print(cmd, "Segments {}-{}", segSt, segSt + segSz - 1);
    Sz5 const shape5{cSz, rSz, tSz * segSz, sSz, vSz};
    Sz6 const shape6{shape[0], shape[1], tps, nSeg, shape[3], shape[4]};
    ks =
      Cx5(ks.reshape(shape6).slice(Sz6{cSt, rSt, tSt, segSt, sSt, vSt}, Sz6{cSz, rSz, tSz, segSz, sSz, vSz}).reshape(shape5));
    Sz3 const shape3{3, rSz, tSz * segSz};
    Sz4 const shape4{3, traj.nSamples(), tps, nSeg};
    traj = Trajectory(traj.points().reshape(shape4).slice(Sz4{0, rSt, tSt, segSt}, Sz4{3, rSz, tSz, segSz}).reshape(shape3),
                      traj.matrix(), traj.voxelSize());
  } else {
    ks = Cx5(ks.slice(Sz5{cSt, rSt, tSt, sSt, vSt}, Sz5{cSz, rSz, tSz, sSz, vSz}));
    traj = Trajectory(traj.points().slice(Sz3{0, rSt, tSt}, Sz3{3, rSz, tSz}), traj.matrix(), traj.voxelSize());
  }

  if (channelStride || sampleStride || traceStride || slabStride || volStride) {
    ks = Cx5(ks.stride(Sz5{channelStride.Get(), sampleStride.Get(), traceStride.Get(), slabStride.Get(), volStride.Get()}));
    traj = Trajectory(traj.points().stride(Sz3{1, sampleStride.Get(), traceStride.Get()}), traj.matrix(), traj.voxelSize());
  }

  HD5::Writer writer(oname.Get());
  writer.writeInfo(info);
  traj.write(writer);
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
  Log::Print(cmd, "Finished");
}
