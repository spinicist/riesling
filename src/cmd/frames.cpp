#include "types.hpp"

#include "basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"

#include <range/v3/numeric.hpp>

int main_frames(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<std::vector<Index>, VectorReader<Index>> frames(
    parser, "F", "List number of traces per frame", {"frames", 'f'});

  ParseCommand(parser);
  if (!oname) {
    throw args::Error("No output filename specified");
  }

  Index const nT = ranges::accumulate(frames.Get(), 0L);
  Index const nF = frames.Get().size();
  rl::Re2 basis(nF, nT);
  basis.setZero();
  Index index = 0;
  for (Index ifr = 0; ifr < nF; ifr++) {
    for (Index it = 0; it < frames.Get()[ifr]; it++) {
      basis(ifr, index++) = 1.;
    }
  }

  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(basis, rl::HD5::Keys::Basis);

  return EXIT_SUCCESS;
}
