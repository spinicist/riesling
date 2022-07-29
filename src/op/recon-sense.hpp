#pragma once

#include "recon.hpp"
#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconSENSE final : ReconOp
{
  using typename ReconOp::Input;
  using typename ReconOp::Output;

  ReconSENSE(GridBase<Cx>*gridder, Cx4 const &maps, SDCOp *sdc = nullptr)
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


  auto A(Input const &x) const -> Output
  {
    Log::Debug("Starting ReconOp forward. Norm {}", Norm(x));
    auto const start = Log::Now();
    auto const y = nufft_.A(sense_.A(x));
    Log::Debug("Finished ReconOp forward. Norm {}. Took {}", Norm(y), Log::ToNow(start));
    return y;
  }

  auto Adj(Output const &x) const -> Input
  {
    Log::Debug("Starting ReconOp adjoint. Norm {}", Norm(x));
    auto const start = Log::Now();
    Input y(inputDimensions());
    y.device(Threads::GlobalDevice()) = sense_.Adj(nufft_.Adj(x));
    Log::Debug("Finished ReconOp adjoint. Norm {}. Took {}.", Norm(y), Log::ToNow(start));
    return y;
  }

  auto AdjA(Input const &x) const -> Input
  {
    Log::Debug("Starting ReconOp adjoint*forward. Norm {}", Norm(x));
    Input y(inputDimensions());
    auto const start = Log::Now();
    y.device(Threads::GlobalDevice()) = sense_.Adj(nufft_.AdjA(sense_.A(x)));
    Log::Debug("Finished ReconOp adjoint*forward. Norm {}. Took {}", Norm(y), Log::ToNow(start));
    return y;
  }

private:
  NUFFTOp nufft_;
  SenseOp sense_;
};

}
