#include "inputs.hpp"
#include "outputs.hpp"

#include "regularizers.hpp"
#include "rl/op/op.hpp"

using namespace rl;

void main_prox(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<float>        ρ(parser, "ρ", "Proximal parameter (1)", {"rho"}, 1.f);
  RegOpts                       regOpts(parser);

  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  HD5::Writer output(oname.Get());
  Cx5 const   x = input.readTensor<Cx5>();
  auto        A = std::make_shared<TOps::Identity<Cx, 5>>(x.dimensions());
  auto [regs, B, ext_x] = Regularizers(regOpts, A);
  Ops::Op<Cx>::Vector xx(B->cols());
  xx.setZero();
  xx.head(A->rows()) = CollapseToConstVector(x);
  for (Index ir = 0; ir < regs.size(); ir++) {
    auto const Fx = regs[ir].T->forward(xx);
    auto const z = regs[ir].P->apply(1.f / ρ.Get(), Fx);
    if (std::holds_alternative<Sz5>(regs[ir].size)) {
      output.writeTensor(fmt::format("prox{:02d}", ir), std::get<Sz5>(regs[ir].size), z.data(), HD5::Dims::Image);
    } else if (std::holds_alternative<Sz6>(regs[ir].size)) {
      output.writeTensor(fmt::format("prox{:02d}", ir), std::get<Sz6>(regs[ir].size), z.data(),
                         {"b", "i", "j", "k", "t", "g"});
    }
  }
  rl::Log::Print(cmd, "Finished");
}
