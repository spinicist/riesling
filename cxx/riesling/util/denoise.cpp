#include "inputs.hpp"
#include "outputs.hpp"
#include "regularizers.hpp"
#include "rl/algo/admm.hpp"
#include "rl/log.hpp"
#include "rl/op/top.hpp"

using namespace rl;

void main_denoise(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  RlsqOpts                      rlsqOpts(parser);
  RegOpts                       regOpts(parser);

  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         in = input.readTensor<Cx5>();
  auto        A = std::make_shared<TOps::Identity<Cx, 5>>(in.dimensions());
  auto [regs, B, ext_x] = Regularizers(regOpts, A);
  Cx5  x(in.dimensions());
  auto xm = CollapseToVector(x);
  if (regs.size() == 1 && !regs[0].T && std::holds_alternative<Sz5>(regs[0].size)) {
    // This regularizer has an analytic solution. Should check ext_x as well but for all current analytic regularizers this will
    // be the identity operator
    regs[0].P->apply(1.f / rlsqOpts.ρ.Get(), CollapseToConstVector(in), xm);
  } else {
    ADMM opt{B,
             nullptr,
             regs,
             rlsqOpts.inner_its0.Get(),
             rlsqOpts.inner_its1.Get(),
             rlsqOpts.atol.Get(),
             rlsqOpts.btol.Get(),
             rlsqOpts.ctol.Get(),
             rlsqOpts.outer_its.Get(),
             rlsqOpts.ε.Get(),
             rlsqOpts.μ.Get(),
             rlsqOpts.τ.Get(),
             nullptr,
             nullptr};
    xm = ext_x->forward(opt.run(CollapseToConstVector(in), rlsqOpts.ρ.Get()));
  }
  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, x.dimensions(), x.data(), HD5::Dims::Image);
  Log::Print(cmd, "Finished");
}
