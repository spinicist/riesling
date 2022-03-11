#pragma once

#include "../cropper.h"
#include "../fft/fft.hpp"
#include "../kernel.h"
#include "../precond/sdc.hpp"
#include "../threads.h"
#include "../trajectory.h"
#include "operator.h"

struct GridBase : Operator<5, 3>
{
  GridBase(Mapping const &map, Index const d1, bool const unsafe)
    : mapping_{map}
    , inputDims_{AddFront(map.cartDims, map.noncartDims[0], d1)}
    , safe_{!unsafe}
    , weightFrames_{true}
  {
  }

  virtual ~GridBase(){};
  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid
  virtual Output A(Input const &cart) const = 0;
  virtual Input Adj(Output const &noncart) const = 0;

  Sz3 outputDimensions() const override
  {
    return this->mapping_.noncartDims;
  }

  Sz5 inputDimensions() const override
  {
    return inputDims_;
  }

  void setUnsafe()
  {
    safe_ = true;
  }

  void setSafe()
  {
    safe_ = false;
  }

  void doNotWeightFrames()
  {
    weightFrames_ = false;
  }

  Mapping const &mapping() const
  {
    return mapping_;
  }

protected:
  Mapping mapping_;
  Sz5 inputDims_;
  bool safe_, weightFrames_;
};

template <int IP, int TP>
struct SizedGrid : GridBase
{
  SizedGrid(SizedKernel<IP, TP> const *k, Mapping const &map, Index const d1, bool const unsafe)
    : GridBase(map, d1, unsafe)
    , kernel_{k}
  {
  }

  virtual ~SizedGrid(){};

  R3 apodization(Sz3 const sz) const
  {
    auto gridSz = this->mapping().cartDims;
    Cx3 temp(gridSz);
    auto const fft = FFT::Make<3, 3>(gridSz);
    temp.setZero();
    auto const k = kernel_->k(Point3{0, 0, 0});
    Crop3(temp, k.dimensions()) = k.template cast<Cx>();
    Log::Image(temp, "apo-kernel");
    fft->reverse(temp);
    R3 a = Crop3(R3(temp.real()), sz);
    float const scale = sqrt(Product(gridSz));
    Log::Print(
      FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    Log::Image(a, "apo-final");
    return a;
  }

protected:
  SizedKernel<IP, TP> const *kernel_;
};