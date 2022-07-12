#pragma once

#include "operator.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconRSSOp final : Operator<4, 3>
{
  ReconRSSOp(GridBase<Cx>*gridder, Sz3 const &dims, SDCOp *sdc = nullptr)
    : nufft_{dims, gridder, sdc}
  {
  }

  InputDims inputDimensions() const
  {
    return LastN<4>(nufft_.inputDimensions());
  }

  OutputDims outputDimensions() const
  {
    return nufft_.outputDimensions();
  }

  template <typename T>
  auto Adj(T const &x) const
  {
    Log::Debug("Starting ReconRSSOp adjoint. Norm {}", Norm(x));
    auto const start = Log::Now();
    Cx5 const channels = nufft_.Adj(x);
    Cx4 y(inputDimensions());
    y.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();
    Log::Debug("Finished ReconOp adjoint. Norm {}. Took {}", Norm(y), Log::ToNow(start));
    return y;
  }

private:
  NUFFTOp nufft_;
};
}
