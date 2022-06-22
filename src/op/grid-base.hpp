#pragma once

#include "../cropper.h"
#include "../fft/fft.hpp"
#include "../kernel.h"
#include "../threads.h"
#include "../trajectory.h"
#include "operator.hpp"

struct GridBase : Operator<5, 3>
{
  GridBase(Mapping const &map, Index const d1, bool const unsafe, std::shared_ptr<Cx5> ws)
    : mapping_{map}
    , inputDims_{AddFront(map.cartDims, map.noncartDims[0], d1)}
    , ws_{ws}
    , safe_{!unsafe}
    , weightFrames_{true}
  {
    if (!ws_) {
      ws_ = std::make_shared<Cx5>(inputDims_);
    } else if (ws_->dimensions() != inputDims_) {
      Log::Fail(FMT_STRING("Workspace dimensions {} did not match input dimensions {}"), ws_->dimensions(), inputDims_);
    }
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
  std::shared_ptr<Cx5> ws_;
  bool safe_, weightFrames_;
};

template <int IP, int TP>
struct SizedGrid : GridBase
{
  SizedGrid(SizedKernel<IP, TP> const *k, Mapping const &map, Index const d1, bool const unsafe, std::shared_ptr<Cx5> ws)
    : GridBase(map, d1, unsafe, ws)
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
    Log::Tensor(temp, "apo-kernel");
    fft->reverse(temp);
    R3 a = Crop3(R3(temp.real()), sz);
    float const scale = sqrt(Product(gridSz));
    Log::Print(FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    Log::Tensor(a, "apo-final");
    return a;
  }

protected:
  SizedKernel<IP, TP> const *kernel_;
};