#include "types.h"

#include "io.h"
#include "log.h"
#include "parse_args.h"

int main_basis_blend(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::Flag mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string> oftype(
    parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");
  args::ValueFlag<long> tp(parser, "TP", "Timepoint within basis for combination", {"tp", 't'}, 0);
  Log log = ParseCommand(parser);

  HD5::Reader input(iname.Get(), log);
  R2 const basis = input.readBasis();
  Cx5 const images = input.readBasisImages();

  if ((tp.Get() < 0) || (tp.Get() >= basis.dimension(0))) {
    Log::Fail(
      FMT_STRING("Requested timepoint {} exceeds basis length {}"), tp.Get(), basis.dimension(0));
  }

  R1 const b = basis.chip(tp.Get(), 0);
  Cx4 const combined =
    (images *
     b.reshape(Sz5{b.dimension(0), 1, 1, 1, 1})
       .broadcast(
         Sz5{1, images.dimension(1), images.dimension(2), images.dimension(3), images.dimension(4)})
       .cast<Cx>())
      .sum(Sz1{0});
  WriteOutput(
    combined, mag, false, input.readInfo(), iname.Get(), oname.Get(), "blend", oftype.Get(), log);
  return EXIT_SUCCESS;
}
