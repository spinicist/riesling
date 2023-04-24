#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

decltype(auto) Blend(Cx4 const &image, Re1 const &b)
{
  Index const x = image.dimension(1);
  Index const y = image.dimension(2);
  Index const z = image.dimension(3);
  Eigen::IndexList<int, FixOne, FixOne, FixOne> rsh;
  rsh.set(0, b.dimension(0));
  Eigen::IndexList<FixOne, int, int, int> brd;
  brd.set(1, x);
  brd.set(2, y);
  brd.set(3, z);
  Eigen::IndexList<FixZero> sum;
  return (image * b.reshape(rsh).broadcast(brd).cast<Cx>()).sum(sum);
}

int main_blend(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::Positional<std::string> bname(parser, "BASIS", "h5 file containing basis");
  args::Flag mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string> oftype(parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> tp(
    parser, "TP", "Timepoints within basis for combination", {"tp", 't'}, {0});
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> keep(parser, "K", "Keep these basis vectors", {"keep", 'k'});
  args::ValueFlag<Index> cycle(parser, "C", "Cycle length", {"cycle"}, -1);

  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 images = input.readTensor<Cx5>(HD5::Keys::Image);
  Sz5 const dims = images.dimensions();

  if (!iname) {
    throw args::Error("No basis file specified");
  }
  HD5::Reader binput(bname.Get());
  Re2 basis = binput.readTensor<Re2>("basis");

  if (basis.dimension(0) != images.dimension(0)) {
    Log::Fail("Basis has {} vectors but image has {}", basis.dimension(1), images.dimension(0));
  }

  if (keep) {
    for (Index ib = 0; ib < basis.dimension(0); ib++) {
      if (std::find(keep.Get().begin(), keep.Get().end(), ib) == keep.Get().end()) {
        basis.chip(ib, 0).setZero();
      }
    }
  }

  auto const &tps = tp.Get();

  Cx5 out(AddFront(LastN<4>(dims), (Index)tps.size()));
  float const scale = std::sqrt(basis.dimension(1));
  for (size_t ii = 0; ii < tps.size(); ii++) {
    Index itp = tps[ii];
    if ((itp < 0) || (itp >= basis.dimension(1))) {
      Log::Fail("Requested timepoint {} exceeds basis length {}", tps[ii], basis.dimension(1));
    }
    Log::Print("Blending timepoint {} cycle {}", itp, cycle.Get());
    while (itp >= tps[ii] && itp < basis.dimension(1)) {
      Re1 const b = basis.chip<1>(itp) * scale;
      for (Index iv = 0; iv < out.dimension(4); iv++) {
        out.chip<4>(iv).chip<0>(ii).device(Threads::GlobalDevice()) += Blend(images.chip<4>(iv), b);
      }
      itp += cycle.Get();
    }
  }

  auto const fname = OutName(iname.Get(), oname.Get(), "blend", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Image, out.dimensions(), out.data());

  return EXIT_SUCCESS;
}
