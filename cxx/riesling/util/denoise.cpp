#include "inputs.hpp"

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
  PDHGArgs                      pdhgArgs(parser);
  args::Flag                    adapt(parser, "A", "Adaptive PDHG", {"adaptive", 'a'});
  args::ValueFlag<std::string>  scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "1");
  RegOpts                       regOpts(parser);
  args::Flag                    residual(parser, "R", "Output residual image", {"resid", 'r'});
  args::Flag                    two(parser, "2", "2x oversample", {"two", '2'});

  ParseCommand(parser, iname, oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader input(iname.Get());
  Cx5         in = input.readTensor<Cx5>();
  float const scale = ScaleImages(scaling.Get(), in);
  if (scale != 1.f) { in.device(Threads::TensorDevice()) = in * Cx(scale); }
  TOps::TOp<5>::Ptr A = two ? MakeOversample(in.dimensions()) : std::make_shared<TOps::Identity<5>>(in.dimensions());
  auto [regs, B, ext_x] = Regularizers(regOpts, A);
  Cx5  x(A->ishape);
  auto xm = CollapseToVector(x);
  if (regs.size() == 1 && !regs[0].T && std::holds_alternative<Sz5>(regs[0].shape)) {
    // This regularizer has an analytic solution. Should check ext_x as well but for all current analytic regularizers this will
    // be the identity operator
    A->adjoint(in, x);
    regs[0].P->apply(1.f, xm);
  } else {
    PDHG::Debug debug = [shape = x.dimensions(), ext_x](Index const ii, PDHG::Vector const &xx, PDHG::Vector const &x̅) {
      if (Log::IsDebugging()) {
        if (ext_x) {
          auto xit = ext_x->forward(xx);
          Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, xit.data(), HD5::Dims::Images);
          xit = ext_x->forward(x̅);
          Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, xit.data(), HD5::Dims::Images);
        } else {
          Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, xx.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, x̅.data(), HD5::Dims::Images);
        }
      }
    };
    if (ext_x) {
      auto xt = PDHG::Run(CollapseToConstVector(in), B, nullptr, regs, pdhgArgs.Get(), debug);
      xm = ext_x->forward(xt);
    } else {
      xm = PDHG::Run(CollapseToConstVector(in), B, nullptr, regs, pdhgArgs.Get(), debug);
    }
  }
  x.device(Threads::TensorDevice()) = x * Cx(1.f / scale);
  HD5::Writer writer(oname.Get());
  auto        info = input.readStruct<Info>(HD5::Keys::Info);
  if (two) { info.voxel_size /= 2; }
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, x.dimensions(), x.data(), HD5::Dims::Images);
  if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
  Log::Print(cmd, "Finished");
}
