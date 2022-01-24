#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "nufft.hpp"
#include "sense.hpp"

struct ReconOp final : Operator<4, 3>
{
  ReconOp(GridBase *gridder, Cx4 const &maps)
    : nufft_{LastN<3>(maps.dimensions()), gridder}
    , sense_{maps, gridder->inputDimensions()[1]}
  {
  }

  InputDims inputDimensions() const
  {
    return sense_.inputDimensions();
  }

  OutputDims outputDimensions() const
  {
    return nufft_.outputDimensions();
  }

  void calcToeplitz() {
    nufft_.calcToeplitz();
  }

  template <typename T>
  auto A(T const &x) const
  {
    Log::Debug("Starting ReconOp forward");
    auto const start = Log::Now();
    auto const result = nufft_.A(sense_.A(x));
    Log::Debug("Finished ReconOp forward: {}", Log::ToNow(start));
    return result;
  }

  template <typename T>
  auto Adj(T const &x) const
  {
    Log::Debug("Starting ReconOp adjoint");
    auto const start = Log::Now();
    auto const result = sense_.Adj(nufft_.Adj(x));
    Log::Debug("Finished ReconOp adjoint: {}", Log::ToNow(start));
    return result;
  }

  template <typename T>
  auto AdjA(T const &x) const
  {
    Log::Debug("Starting ReconOp adjoint*forward");
    auto const start = Log::Now();
    auto const result = sense_.Adj(nufft_.AdjA(sense_.A(x)));
    Log::Debug("Finished ReconOp adjoint*forward: {}", Log::ToNow(start));
    return result;
  }

private:
  NUFFTOp nufft_;
  SenseOp sense_;
};
