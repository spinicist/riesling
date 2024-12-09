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

  RlsqOpts rlsqOpts(parser);
  RegOpts  regOpts(parser);

  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());

  Cx5  in = input.readTensor<Cx5>();
  auto A = std::make_shared<TOps::Identity<Cx, 5>>(in.dimensions());
  auto [reg, B, ext_x] = Regularizers(regOpts, A);
  ADMM opt{B,
           nullptr,
           reg,
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

  auto       x = ext_x->forward(opt.run(CollapseToConstVector(in), rlsqOpts.ρ.Get()));
  auto const xm = AsConstTensorMap(x, in.dimensions());

  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, xm.dimensions(), xm.data(), HD5::Dims::Image);
  Log::Print(cmd, "Finished");
}
