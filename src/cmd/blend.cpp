#include "types.hpp"

#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

#include "tensorOps.hpp"

using namespace rl;

int main_blend(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::Positional<std::string> bname(parser, "BASIS", "h5 file containing basis");
  args::Flag                    mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
  args::ValueFlag<std::string>  oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string>  oftype(parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> sp(parser, "SP", "Samples within basis for combination", {"sp", 's'},
                                                              {0});
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> tp(parser, "TP", "Traces within basis for combination", {"tp", 't'},
                                                              {0});

  ParseCommand(parser);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         images = input.readTensor<Cx5>(HD5::Keys::Image);
  Sz5 const   dims = images.dimensions();

  if (!iname) { throw args::Error("No basis file specified"); }
  auto const basis = ReadBasis(bname.Get());

  if (basis.dimension(0) != images.dimension(0)) {
    Log::Fail("Basis has {} vectors but image has {}", basis.dimension(1), images.dimension(0));
  }

  auto const &tps = tp.Get();

  Cx5         out(AddFront(LastN<4>(dims), (Index)tps.size()));
  float const scale = std::sqrt(basis.dimension(1));
  if (sp && tp && (tp.Get().size() != sp.Get().size())) {
    Log::Fail("Sample points {} and time points {} must be equal if both set", sp.Get().size(), tp.Get().size());
  }
  Basis<Cx>   selected(basis.dimension(0), sp.Get().size(), tp.Get().size());
  for (size_t ii = 0; ii < tp.Get().size(); ii++) {
    Index const is = sp ? sp.Get()[ii] : 0;
    Index const it = tp ? tp.Get()[ii] : 0;
    if ((is < 0) || (is >= basis.dimension(0))) {
      Log::Fail("Requested timepoint {} exceeds samples {}", is, basis.dimension(0));
    }
    if ((it < 0) || (it >= basis.dimension(1))) {
      Log::Fail("Requested timepoint {} exceeds traces {}", it, basis.dimension(1));
    }
    selected.chip<2>(it ? ii : 0).chip<1>(is ? ii : 0) = basis.chip<2>(it).chip<1>(is) * Cx(scale);
  }
  for (Index iv = 0; iv < out.dimension(4); iv++) {
    out.chip<4>(iv).device(Threads::GlobalDevice()) =
      selected.contract(images.chip<4>(iv), Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
  }

  auto const  fname = OutName(iname.Get(), oname.Get(), "blend", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Image, out.dimensions(), out.data());

  return EXIT_SUCCESS;
}
