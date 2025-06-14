#include "inputs.hpp"
#include "outputs.hpp"
#include "regularizers.hpp"
#include "rl/algo/admm.hpp"
#include "rl/algo/pdhg.hpp"
#include "rl/log/log.hpp"
#include "rl/op/top-id.hpp"
#include "rl/scaling.hpp"

using namespace rl;

void main_denoise(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  ADMMArgs                      admmArgs(parser);
  args::Flag                    pdhg(parser, "P", "Use PDHG instead of ADMM (experimental)", {'p', "pdhg"});
  args::ValueFlag<std::string>  scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "otsu");
  args::ValueFlag<Index>        debugIters(parser, "I", "Write debug images ever N outer iterations (16)", {"debug-iters"}, 16);
  args::Flag                    debugZ(parser, "Z", "Write regularizer debug images", {"debug-z"});
  RegOpts                       regOpts(parser);

  ParseCommand(parser, iname, oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader input(iname.Get());
  Cx5         in = input.readTensor<Cx5>();
  float const scale = ScaleImages(scaling.Get(), in);
  if (scale != 1.f) { in.device(Threads::TensorDevice()) = in * Cx(scale); }
  auto A = std::make_shared<TOps::Identity<Cx, 5>>(in.dimensions());
  auto [regs, B, ext_x] = Regularizers(regOpts, A, pdhg);
  Cx5  x(in.dimensions());
  auto xm = CollapseToVector(x);
  if (regs.size() == 1 && !regs[0].T && std::holds_alternative<Sz5>(regs[0].shape)) {
    // This regularizer has an analytic solution. Should check ext_x as well but for all current analytic regularizers this will
    // be the identity operator
    regs[0].P->apply(1.f, CollapseToConstVector(in), xm);
  } else if (pdhg) {
    PDHG::Debug debug = [shape = x.dimensions(), ext_x, regs](Index const ii, PDHG::Vector const &x, PDHG::Vector const &xb,
                                                              PDHG::Vector const &u) {
      if (Log::IsDebugging()) {
        if (ext_x) {
          auto xit = ext_x->forward(x);
          Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, xit.data(), HD5::Dims::Images);
          xit = ext_x->forward(xb);
          Log::Tensor(fmt::format("pdhg-xb-{:02d}", ii), shape, xit.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("pdhg-u-{:02d}", ii), shape, u.data(), HD5::Dims::Images);
        } else {
          Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, x.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("pdhg-xb-{:02d}", ii), shape, xb.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("pdhg-u-{:02d}", ii), shape, u.data(), HD5::Dims::Images);
        }
      }
    };
    PDHG opt{B, nullptr, regs, admmArgs.in_its1.Get(), admmArgs.atol.Get(), 1.f, 16.f, debug};
    xm = ext_x ? ext_x->forward(opt.run(CollapseToConstVector(in))) : opt.run(CollapseToConstVector(in));
  } else {
    ADMM::DebugX debug_x = [shape = x.dimensions(), ext_x, di = debugIters.Get()](Index const ii, ADMM::Vector const &xi) {
      if (ii % di == 0) {
        if (ext_x) {
          auto const xit = ext_x->forward(xi);
          Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, xit.data(), HD5::Dims::Images);
        } else {
          Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, xi.data(), HD5::Dims::Images);
        }
      }
    };
    ADMM::DebugZ debug_z = [&, di = debugIters.Get()](Index const ii, Index const ir, ADMM::Vector const &Fx,
                                                      ADMM::Vector const &z, ADMM::Vector const &u) {
      if (debugZ && (ii % di == 0)) {
        if (std::holds_alternative<Sz5>(regs[ir].shape)) {
          auto const Fshape = std::get<Sz5>(regs[ir].shape);
          Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), HD5::Dims::Images);
        }
        if (std::holds_alternative<Sz6>(regs[ir].shape)) {
          auto const Fshape = std::get<Sz6>(regs[ir].shape);
          Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), {"b", "i", "j", "k", "t", "g"});
          Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), {"b", "i", "j", "k", "t", "g"});
          Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), {"b", "i", "j", "k", "t", "g"});
        }
      }
    };
    ADMM opt{B, nullptr, regs, admmArgs.Get(), debug_x, debug_z};
    xm = ext_x ? ext_x->forward(opt.run(CollapseToConstVector(in))) : opt.run(CollapseToConstVector(in));
  }
  if (scale != 1.f) { x.device(Threads::TensorDevice()) = x / Cx(scale); }
  WriteOutput<5>(cmd, oname.Get(), x, HD5::Dims::Images, input.readStruct<Info>(HD5::Keys::Info));
  Log::Print(cmd, "Finished");
}
