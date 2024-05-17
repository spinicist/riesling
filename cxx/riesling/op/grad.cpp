#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grad.hpp"
#include "op/wavelets.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

void main_grad(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::Flag                    fwd(parser, "F", "Apply forward operation", {"fwd"});
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }

  HD5::Reader reader(iname.Get());
  HD5::Writer writer(oname.Get());
  writer.writeInfo(reader.readInfo());
  if (fwd) {
    auto   input = reader.readTensor<Cx5>();
    Sz4    dims = FirstN<4>(input.dimensions());
    Cx6    output(AddBack(dims, 3, input.dimension(4)));
    TOps::Grad g(dims, std::vector<Index>{1, 2, 3});
    for (Index iv = 0; iv < input.dimension(4); iv++) {
      output.chip<5>(iv) = g.forward(CChipMap(input, iv));
    }
    writer.writeTensor("grad", output.dimensions(), output.data());
  } else {
    auto   input = reader.readTensor<Cx6>("grad");
    Sz4    dims = FirstN<4>(input.dimensions());
    Cx5    output(AddBack(dims, input.dimension(5)));
    TOps::Grad g(dims, std::vector<Index>{1, 2, 3});
    for (Index iv = 0; iv < input.dimension(5); iv++) {
      output.chip<4>(iv) = g.adjoint(CChipMap(input, iv));
    }
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data());
  }
}
