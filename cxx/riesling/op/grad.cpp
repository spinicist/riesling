#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/op/grad.hpp"
#include "rl/op/wavelets.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

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
    auto const    input = reader.readTensor<Cx5>();
    auto const    shape = input.dimensions();
    TOps::Grad<5> g(shape, std::vector<Index>{1, 2, 3});
    auto const    output = g.forward(input);
    writer.writeTensor("data", output.dimensions(), output.data(), {"v", "i", "j", "k", "g", "t"});
  } else {
    auto const    input = reader.readTensor<Cx6>();
    auto const    shape = input.dimensions();
    TOps::Grad<5> g(FirstN<5>(shape), std::vector<Index>{1, 2, 3});
    auto const    output = g.adjoint(input);
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Image);
  }
}
