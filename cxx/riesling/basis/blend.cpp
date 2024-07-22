#include "types.hpp"

#include "basis/basis.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "threads.hpp"

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

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         images = input.readTensor<Cx5>();
  Sz5 const   dims = images.dimensions();

  if (!iname) { throw args::Error("No basis file specified"); }
  Basis const basis(bname.Get());

  if (basis.nV() != images.dimension(0)) {
    Log::Fail("Basis has {} vectors but image has {}", basis.nV(), images.dimension(0));
  }

  auto const &sps = sp.Get();
  auto const &tps = tp.Get();
  if (sps.size() != tps.size()) {
    Log::Fail("Must have same number of trace and sample points");
  }

  Index const ntotal = sps.size() * tps.size();

  Cx5         out(AddFront(LastN<4>(dims), ntotal));
  float const scale = std::sqrt(basis.nSample() * basis.nTrace());

  Cx3 selected(basis.nV(), sps.size(), tps.size());
  for (size_t it = 0; it < tps.size(); it++) {
    if (tps[it] < 0 || tps[it] >= basis.nTrace()) {
      Log::Fail("Requested timepoint {} exceeds traces {}", tps[it], basis.nTrace());
    }

    for (size_t is = 0; is < sps.size(); is++) {
      if (sps[is] < 0 || sps[is] >= basis.nSample()) {
        Log::Fail("Requested timepoint {} exceeds samples {}", sps[is], basis.nSample());
      }
      selected.chip<2>(it).chip<1>(is) = basis.B.chip<2>(tps[it]).chip<1>(sps[is]).conjugate() * Cx(scale);
    }
  }
  Cx2 const sel2 = selected.reshape(Sz2{basis.nV(), ntotal});
  for (Index it = 0; it < out.dimension(4); it++) {
    out.chip<4>(it).device(Threads::GlobalDevice()) =
      sel2.contract(images.chip<4>(it), Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
  }
  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Image);
}
