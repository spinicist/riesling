#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "parse_args.h"
#include "threads.h"

using namespace rl;

decltype(auto) Blend(Cx4 const &image, R1 const &b)
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
  args::ValueFlag<Index> keep(parser, "K", "Keep only N basis vectors", {"keep", 'k'}, 0);

  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 images = input.readTensor<Cx5>(HD5::Keys::Image);
  Sz5 const dims = images.dimensions();
  if (keep) {
    if (keep.Get() < 1 || keep.Get() > dims[0]) {
      Log::Fail(FMT_STRING("Requested to keep {} basis vectors but only {} in file"), keep.Get(), dims[0]);
    }
    Log::Print(FMT_STRING("Keeping {} basis vectors"), keep.Get());
    for (Index ik = keep.Get(); ik < dims[0]; ik++) {
      images.chip(ik, 0).device(Threads::GlobalDevice()) = images.chip(ik, 0).constant(0.f);
    }
  }

  if (!iname) {
    throw args::Error("No basis file specified");
  }
  HD5::Reader binput(bname.Get());
  R2 const basis = binput.readTensor<R2>("basis");

  if (basis.dimension(1) != images.dimension(0)) {
    Log::Fail(FMT_STRING("Basis has {} vectors but image has {}"), basis.dimension(1), images.dimension(0));
  }

  auto const &tps = tp.Get();

  Cx5 out(AddFront(LastN<4>(dims), (Index)tps.size()));
  R1 const scale = basis.chip<0>(0).constant(std::sqrt(basis.dimension(0)));
  for (size_t ii = 0; ii < tps.size(); ii++) {
    if ((tps[ii] < 0) || (tps[ii] >= basis.dimension(0))) {
      Log::Fail(FMT_STRING("Requested timepoint {} exceeds basis length {}"), tps[ii], basis.dimension(0));
    }
    Log::Print(FMT_STRING("Blending timepoint {}"), tps[ii]);
    R1 const b = basis.chip<0>(tps[ii]) * scale;
    for (Index iv = 0; iv < out.dimension(4); iv++) {
      out.chip<4>(iv).chip<0>(ii).device(Threads::GlobalDevice()) = Blend(images.chip<4>(iv), b);
    }
  }

  auto const fname = OutName(iname.Get(), oname.Get(), "blend", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(out, HD5::Keys::Image);

  return EXIT_SUCCESS;
}
