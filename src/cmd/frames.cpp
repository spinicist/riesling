#include "types.hpp"

#include "basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"

#include <range/v3/numeric.hpp>

int main_frames(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> tracesPerFrame(parser, "T", "Traces per frame", {"tpf"});
  args::ValueFlag<Index> framesPerRep(parser, "T", "Frames per repitition", {"fpr"});
  args::ValueFlag<Index> outputFrame(parser, "F", "Only output one frame", {"frame"});

  ParseCommand(parser);
  if (!oname) {
    throw args::Error("No output filename specified");
  }

  Index const nT = tracesPerFrame.Get() * framesPerRep.Get();
  Index const nF = outputFrame ? 1 : framesPerRep.Get();
  Index index = outputFrame ? outputFrame.Get() * tracesPerFrame.Get() : 0;
  rl::Re2 basis(nF, nT);
  basis.setZero();
  for (Index ifr = 0; ifr < nF; ifr++) {
    for (Index it = 0; it < tracesPerFrame.Get(); it++) {
      basis(ifr, index++) = 1.;
    }
  }

  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, basis.dimensions(), basis.data());

  return EXIT_SUCCESS;
}
