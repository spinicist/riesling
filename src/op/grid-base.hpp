#pragma once

#include "../cropper.h"
#include "../fft_plan.h"
#include "../kernel.h"
#include "../trajectory.h"
#include "operator.h"

struct GridBase : Operator<5, 3>
{
  GridBase(Mapping const &map, bool const unsafe)
    : mapping_{map}
    , safe_{!unsafe}
    , weightEchoes_{true}
    , sdcPow_{1.f}

  {
  }

  virtual ~GridBase(){};
  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid
  virtual Output A(Index const nc = 0) const = 0; // Cart k-space must be in workspace()
  virtual void
  Adj(Output const &noncart, Index const nc = 0) const = 0; // Cart k-space -> workspace()

  Sz3 outputDimensions() const override
  {
    return this->mapping_.noncartDims;
  }

  Sz5 inputDimensions() const override
  {
    return workspace_.dimensions();
  }

  void setSDC(float const d)
  {
    std::fill(mapping_.sdc.begin(), mapping_.sdc.end(), d);
  }

  void setSDC(R2 const &sdc)
  {
    std::transform(
      mapping_.noncart.begin(),
      mapping_.noncart.end(),
      mapping_.sdc.begin(),
      [&sdc](NoncartesianIndex const &nc) { return sdc(nc.read, nc.spoke); });
  }

  R2 SDC() const
  {
    R2 sdc(mapping_.noncartDims[1], mapping_.noncartDims[2]);
    sdc.setZero();
    for (size_t ii = 0; ii < mapping_.noncart.size(); ii++) {
      sdc(mapping_.noncart[ii].read, mapping_.noncart[ii].spoke) = mapping_.sdc[ii];
    }
    return sdc;
  }

  void setSDCPower(float const p)
  {
    sdcPow_ = p;
  }

  void setUnsafe()
  {
    safe_ = true;
  }

  void setSafe()
  {
    safe_ = false;
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
  float sdcPow_;
  Cx5 mutable workspace_;
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
    fft.reverse(temp);
    R3 a = Crop3(R3(temp.real()), sz);
    float const scale =
      sqrt(std::accumulate(gridSz.cbegin(), gridSz.cend(), 1, std::multiplies<Index>()));
    Log::Print(
      FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    return a;
  }

protected:
  SizedKernel<IP, TP> const *kernel_;
};