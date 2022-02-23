#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "nufft.hpp"
#include "sense.hpp"

struct ReconRSSOp final : Operator<4, 3>
{
  ReconRSSOp(GridBase *gridder, Sz3 const &dims, Precond *sdc = nullptr)
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
    Log::Debug("Starting ReconRSSOp adjoint");
    auto const start = Log::Now();
    Cx5 channels = nufft_.Adj(x);

    Cx4 image(inputDimensions());
    image.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();

    Log::Debug("Finished ReconOp adjoint: {}", Log::ToNow(start));
    return image;
  }

private:
  NUFFTOp nufft_;
};
