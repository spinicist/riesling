#include "types.hpp"

#include "basis/basis.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "sys/threads.hpp"

#include "tensors.hpp"

using namespace rl;

void main_blend(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> bname(parser, "BASIS", "h5 file containing basis");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::Flag                    mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
  args::ValueFlag<std::string>  oftype(parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> sp(parser, "SP", "Samples within basis for combination", {"sp", 's'},
                                                              {0});
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> tp(parser, "TP", "Traces within basis for combination", {"tp", 't'},
                                                              {0});
  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         images = input.readTensor<Cx5>();
  Sz5 const   dims = images.dimensions();

  if (!iname) { throw args::Error("No basis file specified"); }
  auto const basis = LoadBasis(bname.Get());

  if (basis->nB() != images.dimension(0)) {
    throw Log::Failure(cmd, "Basis has {} vectors but image has {}", basis->nB(), images.dimension(0));
  }

  auto const &sps = sp.Get();
  auto const &tps = tp.Get();
  if (sps.size() != tps.size()) {
    throw Log::Failure(cmd, "Must have same number of trace and sample points");
  }
  Index const nO = sps.size();
  Cx5         out(AddFront(LastN<4>(dims), nO));

  for (Index io = 0; io < nO; io++) {
    out.chip<0>(io).device(Threads::TensorDevice()) =
      basis->blend(images, sps[io], tps[io]);
  }
  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Image);
}
