#pragma once

#include "../cropper.h"
#include "../fft_plan.h"
#include "../kernel.h"
#include "../precond/sdc.hpp"
#include "../trajectory.h"
#include "operator.h"

struct GridBase : Operator<5, 3>
{
  GridBase(Mapping const &map, bool const unsafe)
    : mapping_{map}
    , safe_{!unsafe}
    , weightEchoes_{true}
    , sdc_{nullptr}
  {
  }

  virtual ~GridBase(){};
  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid
  virtual Output A(Index const nc = 0) const = 0; // Cart k-space must be in workspace()
  virtual Input const &
  Adj(Output const &noncart, Index const nc = 0) const = 0; // Cart k-space -> workspace()

  Sz3 outputDimensions() const override
  {
    return this->mapping_.noncartDims;
  }

  Sz5 inputDimensions() const override
  {
    return workspace_.dimensions();
  }

  void setUnsafe()
  {
    safe_ = true;
  }

  void setSafe()
  {
    safe_ = false;
  }

  void setSDC(SDCPrecond const *sdc)
  {
    sdc_ = sdc;
  }

  void doNotWeightEchoes()
  {
    weightEchoes_ = false;
  }

  Mapping const &mapping() const
  {
    return mapping_;
  }

  Cx5 &workspace()
  {
    return workspace_;
  }

protected:
  Mapping mapping_;
  bool safe_, weightEchoes_;
  Cx5 mutable workspace_;
  SDCPrecond const *sdc_;
};

template <int IP, int TP>
struct SizedGrid : GridBase
{
  SizedGrid(SizedKernel<IP, TP> const *k, Mapping const &map, bool const unsafe)
    : GridBase(map, unsafe)
    , kernel_{k}
  {
  }

  virtual ~SizedGrid(){};

  R3 apodization(Sz3 const sz) const
  {
    auto gridSz = this->mapping().cartDims;
    Cx3 temp(gridSz);
    FFT::ThreeD fft(temp);
    temp.setZero();
    auto const k = kernel_->k(Point3{0, 0, 0});
    Crop3(temp, k.dimensions()) = k.template cast<Cx>();
    Log::Image(temp, "apo-kernel.nii");
    fft.reverse(temp);
    R3 a = Crop3(R3(temp.real()), sz);
    float const scale =
      sqrt(std::accumulate(gridSz.cbegin(), gridSz.cend(), 1, std::multiplies<Index>()));
    Log::Print(
      FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    Log::Image(a, "apo-final.nii");
    return a;
  }

protected:
  SizedKernel<IP, TP> const *kernel_;
};