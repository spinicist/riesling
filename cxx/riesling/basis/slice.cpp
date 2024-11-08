#include "types.hpp"

#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "tensors.hpp"

using namespace rl;

void main_basis_slice(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Index> basisStart(parser, "B", "Basis  to start slice", {"basis-start"}, 0);
  args::ValueFlag<Index> basisSize(parser, "B", "Number of basis vectors to keep", {"basis-size"}, 0);
  args::ValueFlag<Index> basisStride(parser, "B", "Basis stride", {"basis-stride"}, 1);

  args::ValueFlag<Index> sampleStart(parser, "S", "Read sample to start slice", {"sample-start"}, 0);
  args::ValueFlag<Index> sampleSize(parser, "S", "Number of read samples to keep", {"sample-size"}, 0);
  args::ValueFlag<Index> sampleStride(parser, "S", "Read sample stride", {"sample-stride"}, 1);

  args::ValueFlag<Index> traceStart(parser, "T", "Trace to start slice", {"trace-start"}, 0);
  args::ValueFlag<Index> traceSize(parser, "T", "Number of traces to keep", {"trace-size"}, 0);
  args::ValueFlag<Index> traceStride(parser, "T", "Trace Stride", {"trace-stride"}, 1);

  args::ValueFlag<Index> tracesPerSeg(parser, "T", "Trace segment size", {"traces-per-seg"}, 0);
  args::ValueFlag<Index> traceSegStart(parser, "T", "Trace segment start", {"seg-start"}, 0);
  args::ValueFlag<Index> traceSegments(parser, "T", "Trace segments", {"seg-size"});

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  auto const  shape = reader.dimensions(HD5::Keys::Basis);
  if (shape.size() != 3) { throw Log::Failure(cmd, "Basis must have 3 dimensions"); }

  Index const bSt = Wrap(basisStart.Get(), shape[0]);
  Index const sSt = Wrap(sampleStart.Get(), shape[1]);
  Index const tSt = Wrap(traceStart.Get(), shape[2]);

  Index const bSz = basisSize ? basisSize.Get() : shape[0] - bSt;
  Index const sSz = sampleSize ? sampleSize.Get() : shape[1] - sSt;
  Index const tSz = traceSize ? traceSize.Get() : (tracesPerSeg ? tracesPerSeg.Get() - tSt : shape[2] - tSt);

  if (bSt + bSz > shape[0]) { throw Log::Failure(cmd, "Last basis {} exceeded maximum {}", bSt + bSz, shape[0]); }
  if (sSt + sSz > shape[1]) { throw Log::Failure(cmd, "Last sample {} exceeded maximum {}", sSt + sSz, shape[1]); }
  if (tracesPerSeg) {
    if (tSt + tSz > tracesPerSeg.Get()) {
      throw Log::Failure(cmd, "Last trace {} exceeded segment size {}", tSt + tSz, tracesPerSeg.Get());
    }
  } else {
    if (tSt + tSz > shape[2]) { throw Log::Failure(cmd, "Last trace {} exceeded maximum {}", tSt + tSz, shape[2]); }
  }

  if (bSz < 1) { throw Log::Failure(cmd, "Basis size was less than 1"); }
  if (sSz < 1) { throw Log::Failure(cmd, "Sample size was less than 1"); }
  if (tSz < 1) { throw Log::Failure(cmd, "Trace size was less than 1"); }

  Log::Print(cmd, "Selected slice {}:{}, {}:{}, {}:{}", bSt, bSt + bSz - 1, sSt, sSt + sSz - 1, tSt, tSt + tSz - 1);

  Cx3 B = reader.readTensor<Cx3>(HD5::Keys::Basis);
  Re1 t = reader.readTensor<Re1>("time");
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
    Sz3 const shape3{bSz, sSz, tSz * segSz};
    Sz4 const shape4{shape[0], shape[1], tps, nSeg};
    B = Cx3(B.reshape(shape4).slice(Sz4{bSt, sSt, tSt, segSt}, Sz4{bSz, sSz, tSz, segSz}).reshape(shape3));
    t = Re1(t.reshape(Sz2{tps, nSeg}).slice(Sz2{tSt, segSt}, Sz2{tSz, segSz}).reshape(Sz1{tSz * segSz}));
  } else {
    B = Cx3(B.slice(Sz3{bSt, sSt, tSt}, Sz3{bSz, sSz, tSz}));
    t = Re1(t.slice(Sz1{tSt}, Sz1{tSz}));
  }

  if (basisStride || sampleStride || traceStride) {
    B = Cx3(B.stride(Sz3{basisStride.Get(), sampleStride.Get(), traceStride.Get()}));
    t = Re1(t.stride(Sz1{traceStride.Get()}));
  }

  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Basis, B.dimensions(), B.data(), HD5::Dims::Basis);
  writer.writeTensor("time", t.dimensions(), t.data(), {"t"});
  Log::Print(cmd, "Finished");
}
