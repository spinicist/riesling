#pragma once

#include "ops.hpp"
#include "tensors.hpp"

namespace rl {

template <typename Scalar_, int InRank_, int OutRank_ = InRank_>
struct TOp : Ops::Op<Scalar_>
{
  using Scalar = Scalar_;
  using Base = Ops::Op<Scalar>;
  static const int InRank = InRank_;
  using InTensor = Eigen::Tensor<Scalar, InRank>;
  using InMap = Eigen::TensorMap<InTensor>;
  using InCMap = Eigen::TensorMap<InTensor const>;
  using InDims = typename InTensor::Dimensions;
  static const int OutRank = OutRank_;
  using OutTensor = Eigen::Tensor<Scalar, OutRank>;
  using OutMap = Eigen::TensorMap<OutTensor>;
  using OutCMap = Eigen::TensorMap<OutTensor const>;
  using OutDims = typename OutTensor::Dimensions;
  using Ptr = std::shared_ptr<TOp<Scalar, InRank, OutRank>>;
  InDims  ishape;
  OutDims oshape;

  TOp(std::string const &n)
    : Ops::Op<Scalar>{n}
  {
    Log::Debug("{} created.", this->name);
  }

  TOp(std::string const &n, InDims const xd, OutDims const yd)
    : Ops::Op<Scalar>{n}
    , ishape{xd}
    , oshape{yd}
  {
    Log::Debug("{} created. Input dims {} Output dims {}", this->name, ishape, oshape);
  }

  virtual ~TOp(){};

  virtual auto rows() const -> Index { return Product(oshape); }
  virtual auto cols() const -> Index { return Product(ishape); }

  using Base::adjoint;
  using Base::forward;
  using Base::inverse;

  void forward(typename Base::CMap const &x, typename Base::Map &y) const final
  {
    assert(x.rows() == this->cols());
    assert(y.rows() == this->rows());
    Log::Debug("Tensor {} forward x {} y {}", this->name, x.rows(), y.rows());
    InCMap xm(x.data(), ishape);
    OutMap ym(y.data(), oshape);
    forward(xm, ym);
  }

  void adjoint(typename Base::CMap const &y, typename Base::Map &x) const final
  {
    assert(x.rows() == this->cols());
    assert(y.rows() == this->rows());
    Log::Debug("Tensor {} adjoint y {} x {}", this->name, y.rows(), x.rows());
    OutCMap ym(y.data(), oshape);
    InMap   xm(x.data(), ishape);
    adjoint(ym, xm);
  }

  virtual auto forward(InTensor const &x) const -> OutTensor
  {
    Log::Debug("Tensor {} forward x {} ishape {} oshape {}", this->name, x.dimensions(), this->ishape, this->oshape);
    InCMap    xm(x.data(), ishape);
    OutTensor y(oshape);
    OutMap    ym(y.data(), oshape);
    forward(xm, ym);
    return y;
  }

  virtual auto adjoint(OutTensor const &y) const -> InTensor
  {
    Log::Debug("Tensor {} adjoint y {} ishape {} oshape {}", this->name, y.dimensions(), this->ishape, this->oshape);
    OutCMap  ym(y.data(), oshape);
    InTensor x(ishape);
    InMap    xm(x.data(), ishape);
    adjoint(ym, xm);
    return x;
  }

  virtual void forward(InCMap const &x, OutMap &y) const = 0;
  virtual void adjoint(OutCMap const &y, InMap &x) const = 0;

  auto startForward(InCMap const &x) const
  {
    if (x.dimensions() != ishape) { Log::Fail("{} forward dims were: {} expected: {}", this->name, x.dimensions(), ishape); }
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Debug("{} forward started. Dimensions {}->{}. Norm {}", this->name, this->ishape, this->oshape, Norm(x));
    }
    return Log::Now();
  }

  void finishForward(OutMap const &y, Log::Time const start) const
  {
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Debug("{} forward finished. Took {}. Norm {}.", this->name, Log::ToNow(start), Norm(y));
    }
  }

  auto startAdjoint(OutCMap const &y) const
  {
    if (y.dimensions() != oshape) { Log::Fail("{} adjoint dims were: {} expected: {}", this->name, y.dimensions(), oshape); }
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Debug("{} adjoint started. Dimensions {}->{}. Norm {}", this->name, this->oshape, this->ishape, Norm(y));
    }
    return Log::Now();
  }

  void finishAdjoint(InMap const &x, Log::Time const start) const
  {
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Debug("{} adjoint finished. Took {}. Norm {}", this->name, Log::ToNow(start), Norm(x));
    }
  }
};

#define OP_INHERIT(SCALAR, INRANK, OUTRANK)                                                                                    \
  using Parent = TOp<SCALAR, INRANK, OUTRANK>;                                                                      \
  using Scalar = typename Parent::Scalar;                                                                                      \
  static const int InRank = Parent::InRank;                                                                                    \
  using InTensor = typename Parent::InTensor;                                                                                  \
  using InMap = typename Parent::InMap;                                                                                        \
  using InCMap = typename Parent::InCMap;                                                                                      \
  using InDims = typename Parent::InDims;                                                                                      \
  using Parent::ishape;                                                                                                        \
  static const int OutRank = Parent::OutRank;                                                                                  \
  using OutTensor = typename Parent::OutTensor;                                                                                \
  using OutMap = typename Parent::OutMap;                                                                                      \
  using OutCMap = typename Parent::OutCMap;                                                                                    \
  using OutDims = typename Parent::OutDims;                                                                                    \
  using Parent::oshape;

#define OP_DECLARE(SELF)                                                                                                       \
  using Ptr = std::shared_ptr<SELF>;                                                                                           \
  void forward(InCMap const &x, OutMap &y) const;                                                                              \
  void adjoint(OutCMap const &y, InMap &x) const;                                                                              \
  using Parent::forward;                                                                                                       \
  using Parent::adjoint;

template <typename Scalar_, int Rank>
struct TensorIdentity : TOp<Scalar_, Rank, Rank>
{
  OP_INHERIT(Scalar_, Rank, Rank)
  TensorIdentity(Sz<Rank> dims)
    : Parent("Identity", dims, dims)
  {
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const time = Parent::startForward(x);
    y = x;
    Parent::finishAdjoint(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    auto const time = Parent::startAdjoint(y);
    x = y;
    Parent::finishAdjoint(x, time);
  }
};

} // namespace rl
