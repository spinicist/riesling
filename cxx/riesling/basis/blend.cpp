#include "rl/basis/basis.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include "inputs.hpp"

using namespace rl;

void main_blend(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> bname(parser, "BASIS", "h5 file containing basis");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::Flag                    mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
  args::ValueFlag<std::string>  dset(parser, "D", "Input dataset", {"dset"}, "data");
  args::ValueFlag<Index>        nr(parser, "N", "Retain N basis vectors", {"nr"});
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> sp(parser, "SP", "Samples within basis for combination", {"sp", 's'},
                                                              {0});
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> tp(parser, "TP", "Traces within basis for combination", {"tp", 't'},
                                                              {0});
  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         images = input.readTensor<Cx5>(dset.Get());
  Sz5 const   ishape = images.dimensions();

  if (!iname) { throw args::Error("No basis file specified"); }
  auto const basis = LoadBasis(bname.Get());

  if (basis->nB() != images.dimension(3)) {
    throw Log::Failure(cmd, "Basis has {} vectors but image has {}", basis->nB(), images.dimension(3));
  }

  auto sps = sp.Get();
  auto tps = tp.Get();
  if (sps.size() == 1) { sps.resize(tps.size()); }
  if (sps.size() != tps.size()) { throw Log::Failure(cmd, "Must have same number of trace and sample points"); }
  Index const nO = sps.size();
  Index const nT = ishape[4];
  Cx5         out(AddBack(FirstN<3>(ishape), nO, nT));

  for (Index io = 0; io < nO; io++) {
    out.chip<3>(io).device(Threads::TensorDevice()) = basis->blend(images, sps[io], tps[io], nr.Get());
  }

  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, input.readStruct<Info>(HD5::Keys::Info));
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Images);
}
