#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grad.hpp"
#include "op/wavelets.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

int main_grad(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input image file");
  args::ValueFlag<std::string>  oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::Flag                    fwd(parser, "F", "Apply forward operation", {"fwd"});
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }

  HD5::Reader reader(iname.Get());
  auto const  fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(reader.readInfo());

  if (fwd) {
    auto   input = reader.readTensor<Cx5>(HD5::Keys::Image);
    Sz4    dims = FirstN<4>(input.dimensions());
    Cx6    output(AddBack(dims, 3, input.dimension(4)));
    GradOp g(dims, std::vector<Index>{1, 2, 3});
    for (Index iv = 0; iv < input.dimension(4); iv++) {
      output.chip<5>(iv) = g.forward(CChipMap(input, iv));
    }
    writer.writeTensor("grad", output.dimensions(), output.data());
  } else {
    auto   input = reader.readTensor<Cx6>("grad");
    Sz4    dims = FirstN<4>(input.dimensions());
    Cx5    output(AddBack(dims, input.dimension(5)));
    GradOp g(dims, std::vector<Index>{1, 2, 3});
    for (Index iv = 0; iv < input.dimension(5); iv++) {
      output.chip<4>(iv) = g.adjoint(CChipMap(input, iv));
    }
    writer.writeTensor(HD5::Keys::Image, output.dimensions(), output.data());
  }

  return EXIT_SUCCESS;
}
