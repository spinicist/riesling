#include "sdc.hpp"

#include "threads.hpp"

namespace rl {

SDCOp::SDCOp(Re2 const &dc, Index const nc)
  : dims_{AddFront(dc.dimensions(), nc)}
  , dc_{dc}
{
}

SDCOp::SDCOp(Sz2 const &dims, Index const nc)
  : dims_{AddFront(dims, nc)}
{
}

auto SDCOp::inputDimensions() const -> InputDims
{
  return dims_;
}

auto SDCOp::outputDimensions() const -> OutputDims
{
  return dims_;
}

auto SDCOp::adjoint(Output const &x) const -> Input
{
  checkOutput(x.dimensions());
  if (dc_.size()) {
    auto const start = Log::Now();
    auto const dims = x.dimensions();
    Cx3 p(dims);
    p.device(Threads::GlobalDevice()) =
      x * dc_.cast<Cx>().reshape(Sz3{1, dc_.dimension(0), dc_.dimension(1)}).broadcast(Sz3{dims[0], 1, 1});
    Log::Debug(FMT_STRING("SDC Adjoint Took {}"), Log::ToNow(start));
    return p;
  } else {
    Log::Debug(FMT_STRING("No SDC"));
    return x;
  }
}

} // namespace rl
