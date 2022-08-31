#pragma once

#include "operator.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconOp final : Operator<4, 3>
{
  ReconOp(GridBase<Cx, 3>*gridder, Cx4 const &maps, SDCOp *sdc = nullptr)
    : nufft_{LastN<3>(maps.dimensions()), gridder, sdc}
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

  void calcToeplitz()
  {
    nufft_.calcToeplitz();
  }

  template <typename T>
  auto forward(T const &x) const
  {
    Log::Debug("Starting ReconOp forward. Norm {}", Norm(x));
    auto const start = Log::Now();
    auto const y = nufft_.forward(sense_.forward(x));
    Log::Debug("Finished ReconOp forward. Norm {}. Took {}", Norm(y), Log::ToNow(start));
    return y;
  }

  template <typename T>
  auto adjoint(T const &x) const
  {
    Log::Debug("Starting ReconOp adjoint. Norm {}", Norm(x));
    auto const start = Log::Now();
    Input y(inputDimensions());
    y.device(Threads::GlobalDevice()) = sense_.adjoint(nufft_.adjoint(x));
    Log::Debug("Finished ReconOp adjoint. Norm {}. Took {}.", Norm(y), Log::ToNow(start));
    return y;
  }

  template <typename T>
  Input adjfwd(T const &x) const
  {
    Log::Debug("Starting ReconOp adjoint*forward. Norm {}", Norm(x));
    Input y(inputDimensions());
    auto const start = Log::Now();
    y.device(Threads::GlobalDevice()) = sense_.adjoint(nufft_.adjfwd(sense_.forward(x)));
    Log::Debug("Finished ReconOp adjoint*forward. Norm {}. Took {}", Norm(y), Log::ToNow(start));
    return y;
  }

private:
  NUFFTOp nufft_;
  SenseOp sense_;
};
}
