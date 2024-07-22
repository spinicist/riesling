#include "types.hpp"

#include "basis/basis.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "threads.hpp"

#include "tensors.hpp"

using namespace rl;

void main_basis_R(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> bname(parser, "BASIS", "h5 file containing basis");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  ParseCommand(parser);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         images = input.readTensor<Cx5>();
  Sz5 const   dims = images.dimensions();

  if (!iname) { throw args::Error("No basis file specified"); }
  auto const basis = std::make_shared<Basis>(bname.Get());

  if (basis->nV() != images.dimension(0)) {
    Log::Fail("Basis has {} vectors but image has {}", basis->nV(), images.dimension(0));
  }

  basis->applyR(images);


  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Image);
}
