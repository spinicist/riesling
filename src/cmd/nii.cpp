#include "types.h"

#include "io/hd5.hpp"
#include "io/nifti.hpp"
#include "log.h"
#include "parse_args.h"

int main_nii(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input h5 file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output nii file");
  args::Flag mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
  args::ValueFlag<std::string> dset(parser, "D", "Dataset name (default image)", {'d', "dset"}, "image");
  args::ValueFlag<Index> frameArg(parser, "E", "Frame (default all)", {'e', "frame"}, 0);
  args::ValueFlag<Index> volArg(parser, "V", "Volume (default all)", {"volume"}, 0);
  ParseCommand(parser, iname);
  if (!oname) {
    Log::Fail("No output name specified");
  }
  HD5::Reader input(iname.Get());
  Info const info = input.readInfo();
  Cx5 const image = input.readTensor<Cx5>(dset.Get());
  Sz5 const sz = image.dimensions();

  Index const szE = frameArg ? 1 : sz[0];
  Index const stE = frameArg ? 0 : frameArg.Get();
  Index const szV = volArg ? 1 : sz[4];
  Index const stV = volArg ? 0 : volArg.Get();

  if (stE < 0 || stE >= image.dimension(0)) {
    Log::Fail(FMT_STRING("Requested frame {} is out of range 0-{}"), stE, sz[0]);
  }
  if (stV < 0 || stE >= image.dimension(4)) {
    Log::Fail(FMT_STRING("Requested frame {} is out of range 0-{}"), stE, sz[4]);
  }

  Cx4 const output = image.slice(Sz5{stE, 0, 0, 0, stV}, Sz5{szE, sz[1], sz[2], sz[3], szV})
                       .shuffle(Sz5{1, 2, 3, 0, 4})
                       .reshape(Sz4{sz[1], sz[2], sz[3], szE * szV});
  if (mag) {
    WriteNifti(info, R4(output.abs()), oname.Get());
  } else {
    WriteNifti(info, output, oname.Get());
  }

  return EXIT_SUCCESS;
}
