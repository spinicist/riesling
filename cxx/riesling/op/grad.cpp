#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/grad.hpp"
#include "rl/op/laplacian.hpp"
#include "rl/op/wavelets.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_grad(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::Flag                    fwd(parser, "F", "Apply forward operation", {"fwd"});
  args::Flag                    div(parser, "D", "Apply Div", {"div"});
  args::Flag                    vec(parser, "V", "Apply Vector Gradient", {"vec"});
  args::Flag                    lap(parser, "F", "Apply Laplacian", {"lap", 'l'});
  args::Flag                    lap2(parser, "F", "Apply isotropic Laplacian", {"lap2"});
  args::ValueFlag<int>          diffOrder(parser, "G", "Finite difference scheme", {"diff"}, 0);
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }

  HD5::Reader reader(iname.Get());
  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info));
  if (fwd) {
    if (lap) {
      auto const         input = reader.readTensor<Cx5>();
      auto const         shape = input.dimensions();
      TOps::Laplacian<5> g(shape);
      auto const         output = g.forward(input);
      writer.writeTensor("data", output.dimensions(), output.data(), {"i", "j", "k", "b", "t"});
    } else if (lap2) {
      auto const      input = reader.readTensor<Cx5>();
      auto const      shape = input.dimensions();
      TOps::IsoΔ3D<5> g(shape);
      auto const      output = g.forward(input);
      writer.writeTensor("data", output.dimensions(), output.data(), {"i", "j", "k", "b", "t"});
    } else if (div) {
      auto const      input = reader.readTensor<Cx6>();
      auto const      shape = input.dimensions();
      TOps::Div<5, 3> g(FirstN<5>(shape), Sz3{0, 1, 2});
      auto const      output = g.forward(input);
      writer.writeTensor("data", output.dimensions(), output.data(), {"i", "j", "k", "b", "t"});
    } else if (vec) {
      auto const          input = reader.readTensor<Cx6>();
      auto const          shape = input.dimensions();
      TOps::GradVec<6, 3> g(shape, Sz3{0, 1, 2});
      auto const          output = g.forward(input);
      writer.writeTensor("data", output.dimensions(), output.data(), {"i", "j", "k", "b", "t", "g"});
    } else {
      auto const       input = reader.readTensor<Cx5>();
      auto const       shape = input.dimensions();
      TOps::Grad<5, 3> g(shape, Sz3{0, 1, 2});
      auto const       output = g.forward(input);
      writer.writeTensor("data", output.dimensions(), output.data(), {"i", "j", "k", "b", "t", "g"});
    }
  } else {
    if (lap) {
      auto const         input = reader.readTensor<Cx5>();
      auto const         shape = input.dimensions();
      TOps::Laplacian<5> g(shape);
      auto const         output = g.adjoint(input);
      writer.writeTensor("data", output.dimensions(), output.data(), {"i", "j", "k", "b", "t"});
    } else if (lap2) {
      auto const      input = reader.readTensor<Cx5>();
      auto const      shape = input.dimensions();
      TOps::IsoΔ3D<5> g(shape);
      auto const      output = g.adjoint(input);
      writer.writeTensor("data", output.dimensions(), output.data(), {"i", "j", "k", "b", "t"});
    } else if (div) {
      auto const       input = reader.readTensor<Cx5>();
      auto const       shape = input.dimensions();
      TOps::Grad<5, 3> g(shape, Sz3{0, 1, 2});
      auto const       output = g.forward(input);
      writer.writeTensor("data", output.dimensions(), output.data(), {"i", "j", "k", "b", "t", "g"});
    } else if (vec) {
      auto const          input = reader.readTensor<Cx6>();
      auto const          shape = input.dimensions();
      TOps::GradVec<6, 3> g(AddBack(FirstN<5>(shape), 3), Sz3{0, 1, 2});
      auto const          output = g.adjoint(input);
      writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), {"i", "j", "k", "b", "t", "g"});
    } else {
      auto const       input = reader.readTensor<Cx6>();
      auto const       shape = input.dimensions();
      TOps::Grad<5, 3> g(FirstN<5>(shape), Sz3{0, 1, 2});
      auto const       output = g.adjoint(input);
      writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Images);
    }
  }
}
