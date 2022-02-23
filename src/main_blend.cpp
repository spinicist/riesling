#include "types.h"

#include "io.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"

decltype(auto) Blend(Cx5 const &images, R1 const &b)
{
  Index const x = images.dimension(1);
  Index const y = images.dimension(2);
  Index const z = images.dimension(3);
  Index const v = images.dimension(4);
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> rsh;
  rsh.set(0, b.dimension(0));
  Eigen::IndexList<FixOne, int, int, int, int> brd;
  brd.set(1, x);
  brd.set(2, y);
  brd.set(3, z);
  brd.set(4, v);
  Eigen::IndexList<FixZero> sum;
  return (images * b.reshape(rsh).broadcast(brd).cast<Cx>()).sum(sum);
}

int main_blend(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::Positional<std::string> bname(parser, "BASIS", "h5 file containing basis");
  args::Flag mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string> oftype(
    parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> tp(
    parser, "TP", "Timepoints within basis for combination", {"tp", 't'}, {0});
  args::Flag eddy_rss(
    parser, "", "Produce an RSS image for eddy-current correction", {"eddy", 'e'});

  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 const images = input.readTensor<Cx5>("image");
  Sz5 const dims = images.dimensions();
  if (!iname) {
    throw args::Error("No basis file specified");
  }
  HD5::Reader binput(bname.Get());
  R2 const basis = binput.readTensor<R2>("basis");

  if (basis.dimension(1) != images.dimension(0)) {
    Log::Fail(
      FMT_STRING("Basis has {} vectors but image has {}"), basis.dimension(1), images.dimension(0));
  }

  auto const &tps = tp.Get();

  Cx5 out(AddFront(LastN<4>(dims), tps.size()));

  for (size_t ii = 0; ii < tps.size(); ii++) {
    if ((tps[ii] < 0) || (tps[ii] >= basis.dimension(0))) {
      Log::Fail(
        FMT_STRING("Requested timepoint {} exceeds basis length {}"), tps[ii], basis.dimension(0));
    }
    Log::Print(FMT_STRING("Blending timepoint {}"), tps[ii]);
    R1 const b = basis.chip<0>(tps[ii]);
    out.chip<0>(ii).device(Threads::GlobalDevice()) = Blend(images, b);
  }

  auto const fname = OutName(iname.Get(), oname.Get(), "blend", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(out, "image");

  return EXIT_SUCCESS;
}
