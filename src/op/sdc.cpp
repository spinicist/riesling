#include "sdc.hpp"
#include "../sdc.hpp"

#include "identity.hpp"
#include "threads.hpp"

namespace rl {

SDCOp::SDCOp(Re2 const &dc, Index const nc)
  : dims_{AddFront(dc.dimensions(), nc)}
  , dc_{dc}
  , ws_{dims_}
{
}

auto SDCOp::inputDimensions() const -> InputDims { return dims_; }

auto SDCOp::outputDimensions() const -> OutputDims { return dims_; }

auto SDCOp::forward(Input const &x) const -> Output const & {
  ws_ = x;
  return ws_;
}

auto SDCOp::adjoint(Output const &y) const -> Input const &
{
  assert(y.dimensions() == outputDimensions());
  Log::Print<Log::Level::Debug>(FMT_STRING("SDC Adjoint"));
  auto const start = Log::Now();
  ws_ =
    y * dc_.cast<Cx>().reshape(Sz3{1, dc_.dimension(0), dc_.dimension(1)}).broadcast(Sz3{dims_[0], 1, 1});
  LOG_DEBUG(FMT_STRING("SDC Adjoint Took {}"), Log::ToNow(start));
  return ws_;
}

} // namespace rl
