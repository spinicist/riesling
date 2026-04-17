#include "args/all.hpp"
#include "args/admm.hpp"

#include "regularizers.hpp"
#include "rl/algo/admm.hpp"
#include "rl/algo/pdhg.hpp"
#include "rl/log/debug.hpp"
#include "rl/log/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/pad.hpp"
#include "rl/op/top-id.hpp"
#include "rl/scaling.hpp"

using namespace rl;

auto MakeOversample(Sz5 const shape) -> TOps::TOp<5>::Ptr
{
  Sz5 const         shape2{shape[0] * 2, shape[1] * 2, shape[2] * 2, shape[3], shape[4]};
  TOps::TOp<5>::Ptr FFT1 = TOps::FFT<5, 3>::Make(shape, true);
  TOps::TOp<5>::Ptr FFT2 = TOps::FFT<5, 3>::Make(shape2);
  TOps::TOp<5>::Ptr crop = TOps::Crop<5>::Make(shape2, shape, std::sqrt(1.f/8.f));
  TOps::TOp<5>::Ptr fc = TOps::MakeCompose(FFT2, crop);
  TOps::TOp<5>::Ptr fcf = TOps::MakeCompose(fc, FFT1);
  return fcf;
}

void main_denoise(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  ADMMArgs                      admmArgs(parser);
  PDHGArgs                      pdhgArgs(parser);
	args::Flag                    pdhg(parser, "P", "Use PDHG instead of ADMM", {"pdhg", 'p'});
  args::Flag                    adapt(parser, "A", "Adaptive PDHG", {"adaptive", 'a'});
  args::ValueFlag<std::string>  scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "1");
  RegOpts                       regOpts(parser);
  args::Flag                    residual(parser, "R", "Output residual image", {"resid", 'r'});
  args::Flag                    two(parser, "2", "2x oversample", {"two", '2'});
  args::ValueFlag<Index>        debugIters(parser, "I", "Write debug images ever N outer iterations (1)", {"debug-iters"}, 1);

  ParseCommand(parser, iname, oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader input(iname.Get());
  Cx5         in = input.readTensor<Cx5>();
  float const scale = ScaleImages(scaling.Get(), in);
  if (scale != 1.f) { in.device(Threads::TensorDevice()) = in / Cx(scale); }
  TOps::TOp<5>::Ptr E = two ? MakeOversample(in.dimensions()) : std::make_shared<TOps::Identity<5>>(in.dimensions());
  auto [regs, A, ext_x] = Regularizers(regOpts, E);
  VectorX x(A->cols());
  if (pdhg) {
    PDHG::Debug debug = [shape = E->ishape](Index const ii, PDHG::Vector const &x, PDHG::Vector const &x̅) {
      if (Log::IsDebugging()) {
	      Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, x.data(), HD5::Dims::Images);
	      Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, x̅.data(), HD5::Dims::Images);
      }
    };
    x = PDHG::Run(CollapseToConstVector(in), A, nullptr, regs, pdhgArgs.Get(), debug);
  } else {
    ADMM::DebugX debug_x = [shape=E->ishape, di = debugIters.Get(), ext_x](Index const ii, ADMM::Vector const &x) {
      if (Log::IsDebugging() && (ii % di == 0)) {
	   		Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data(), HD5::Dims::Images);
      }
    };
    ADMM::DebugZ debug_z = [&, di = debugIters.Get()](Index const ii, Index const ir, ADMM::Vector const &Fx,
                                                      ADMM::Vector const &z, ADMM::Vector const &u) {
      if (Log::IsDebugging() && (ii % di == 0)) {
        if (std::holds_alternative<Sz5>(regs[ir].shape)) {
          auto const Fshape = std::get<Sz5>(regs[ir].shape);
          Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), HD5::Dims::Images);
        }
        if (std::holds_alternative<Sz6>(regs[ir].shape)) {
          auto const Fshape = std::get<Sz6>(regs[ir].shape);
          Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), {"i", "j", "k", "b",  "t", "g"});
          Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), {"i", "j", "k", "b", "t", "g"});
          Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), {"i", "j", "k", "b", "t", "g"});
        }
      }
    };
    ADMM opt{A, nullptr, regs, admmArgs.Get(), debug_x, debug_z};
   	x = opt.run(CollapseToConstVector(in));
  }

  x.device(Threads::TensorDevice()) = x * scale;
  HD5::Writer writer(oname.Get());
  auto        info = input.readStruct<Info>(HD5::Keys::Info);
  if (two) { info.voxel_size /= 2; }
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, in.dimensions(), x.data(), HD5::Dims::Images);
  if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
  Log::Print(cmd, "Finished");
}
