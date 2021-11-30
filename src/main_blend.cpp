#include "types.h"

#include "io.h"
#include "log.h"
#include "parse_args.h"

decltype(auto) Blend(Cx5 const &images, R1 const &b)
{
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> rsh;
  rsh.set(0, b.dimension(0));
  Eigen::IndexList<FixOne, int, int, int, int> brd;
  brd.set(1, images.dimension(1));
  brd.set(2, images.dimension(2));
  brd.set(3, images.dimension(3));
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
  args::ValueFlag<long> tp(parser, "TP", "Timepoint within basis for combination", {"tp", 't'}, 0);
  args::Flag eddy_rss(
    parser, "", "Produce an RSS image for eddy-current correction", {"eddy", 'e'});

  Log log = ParseCommand(parser);

  HD5::Reader input(iname.Get(), log);
  Cx5 const images = input.readBasisImages();
  HD5::Reader binput(bname.Get(), log);
  R2 const basis = binput.readBasis();

  if ((tp.Get() < 0) || (tp.Get() >= basis.dimension(0))) {
    Log::Fail(
      FMT_STRING("Requested timepoint {} exceeds basis length {}"), tp.Get(), basis.dimension(0));
  }

  if (eddy_rss) {
    R1 const b0 = basis.chip<0>(0);
    R1 const b1 = basis.chip<0>(tp.Get());
    R1 const b2 = basis.chip<0>(2 * tp.Get());
    R1 const b3 = basis.chip<0>(3 * tp.Get());
    Cx4 const rss = ((Blend(images, b2) - Blend(images, b0)).square() +
                     (Blend(images, b3) - Blend(images, b1)).square())
                      .sqrt();
    WriteOutput(rss, mag, false, input.readInfo(), iname.Get(), oname.Get(), "blend", "h5", log);
  } else {
    R1 const b = basis.chip<0>(tp.Get());
    Cx4 const combined = Blend(images, b);
    WriteOutput(
      combined, mag, false, input.readInfo(), iname.Get(), oname.Get(), "blend", "h5", log);
  }
  return EXIT_SUCCESS;
}
