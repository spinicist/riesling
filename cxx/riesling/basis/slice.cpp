#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include "inputs.hpp"

using namespace rl;

void main_basis_slice(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  SzFlag<3>              basis(parser, "BASIS", "Basis start,size,stride", {"basis"}, Sz3{0, 0, 1});
  SzFlag<3>              sample(parser, "SAMPLE", "Sample start,size,stride", {"sample"}, Sz3{0, 0, 1});
  SzFlag<3>              trace(parser, "TRACE", "Trace start,size,stride", {"trace"}, Sz3{0, 0, 1});
  SzFlag<3>              segment(parser, "SEG", "Segment start,size,stride", {"segment"}, Sz3{0, 0, 1});
  args::ValueFlag<Index> tps(parser, "SEG", "Traces per segment", {"tps"}, 0);

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  auto const  shape = reader.dimensions(HD5::Keys::Basis);
  if (shape.size() != 3) { throw Log::Failure(cmd, "Basis must have 3 dimensions"); }

  Index const bSt = Wrap(basis.Get()[0], shape[0]);
  Index const sSt = Wrap(sample.Get()[0], shape[1]);
  Index const tSt = Wrap(trace.Get()[0], shape[2]);

  Index const bSz = (basis.Get()[1] > 0) ? basis.Get()[1] : shape[0] - bSt;
  Index const sSz = (sample.Get()[1] > 0) ? sample.Get()[1] : shape[1] - sSt;
  Index const tSz = (trace.Get()[1] > 0) ? trace.Get()[1] : (tps ? tps.Get() - tSt : shape[2] - tSt);

  if (bSt + bSz > shape[0]) { throw Log::Failure(cmd, "Last basis {} exceeded maximum {}", bSt + bSz, shape[0]); }
  if (sSt + sSz > shape[1]) { throw Log::Failure(cmd, "Last sample {} exceeded maximum {}", sSt + sSz, shape[1]); }
  if (tps) {
    if (tSt + tSz > tps.Get()) { throw Log::Failure(cmd, "Last trace {} exceeded segment size {}", tSt + tSz, tps.Get()); }
  } else {
    if (tSt + tSz > shape[2]) { throw Log::Failure(cmd, "Last trace {} exceeded maximum {}", tSt + tSz, shape[2]); }
  }

  if (bSz < 1) { throw Log::Failure(cmd, "Basis size was less than 1"); }
  if (sSz < 1) { throw Log::Failure(cmd, "Sample size was less than 1"); }
  if (tSz < 1) { throw Log::Failure(cmd, "Trace size was less than 1"); }

  Log::Print(cmd, "Selected slice {}:{}, {}:{}, {}:{}", bSt, bSt + bSz - 1, sSt, sSt + sSz - 1, tSt, tSt + tSz - 1);

  Cx3 B = reader.readTensor<Cx3>(HD5::Keys::Basis);
  if (tps) {
    if (tSt + tSz > tps.Get()) { throw Log::Failure(cmd, "Selected traces {}-{} extend past segment {}", tSt, tSz, tps.Get()); }
    Index const nSeg = shape[2] / tps.Get();
    if (nSeg * tps.Get() != shape[2]) {
      throw Log::Failure(cmd, "Traces per seg {} does not cleanly divide traces {}", tps.Get(), shape[2]);
    }
    Index const segSt = Wrap(segment.Get()[0], nSeg);
    Index const segSz = segment.Get()[1] ? std::clamp(segment.Get()[1], 1L, nSeg) : nSeg - segSt;
    Index const segStride = segment.Get()[2] > 0 ? segment.Get()[2] : 1;
    Log::Print(cmd, "Segments {}-{}", segSt, segSt + segSz - 1);
    Sz3 const shape3{bSz, sSz, tSz * segSz};
    Sz4 const shape4{shape[0], shape[1], tps.Get(), nSeg};
    if (segStride > 1) {
      B = Cx3(B.reshape(shape4)
                .slice(Sz4{bSt, sSt, tSt, segSt}, Sz4{bSz, sSz, tSz, segSz})
                .stride(Sz4{1, 1, 1, segStride})
                .reshape(shape3));
    } else {
      B = Cx3(B.reshape(shape4).slice(Sz4{bSt, sSt, tSt, segSt}, Sz4{bSz, sSz, tSz, segSz}).reshape(shape3));
    }
  } else {
    B = Cx3(B.slice(Sz3{bSt, sSt, tSt}, Sz3{bSz, sSz, tSz}));
  }

  B = Cx3(B.stride(Sz3{basis.Get()[2], sample.Get()[2], trace.Get()[2]}));
  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Basis, B.dimensions(), B.data(), HD5::Dims::Basis);

  Log::Print(cmd, "Finished");
}
