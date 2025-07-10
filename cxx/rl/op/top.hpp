#pragma once

#include "../tensors.hpp"
#include "ops.hpp"

namespace rl::TOps {

template <int InRank_, int OutRank_ = InRank_> struct TOp : Ops::Op
{
  using Base = Ops::Op;
  static const int InRank = InRank_;
  using InTensor = Eigen::Tensor<Cx, InRank>;
  using InMap = Eigen::TensorMap<InTensor>;
  using InCMap = Eigen::TensorMap<InTensor const>;
  using InDims = typename InTensor::Dimensions;
  static const int OutRank = OutRank_;
  using OutTensor = Eigen::Tensor<Cx, OutRank>;
  using OutMap = Eigen::TensorMap<OutTensor>;
  using OutCMap = Eigen::TensorMap<OutTensor const>;
  using OutDims = typename OutTensor::Dimensions;
  using Ptr = std::shared_ptr<TOp<InRank, OutRank>>;
  using Time = Base::Time;
  InDims  ishape;
  OutDims oshape;

  TOp(std::string const &n);
  TOp(std::string const &n, InDims const xd, OutDims const yd);

  virtual ~TOp() = default;

  auto rows() const -> Index final;
  auto cols() const -> Index final;

  using Base::adjoint;
  using Base::forward;
  using Base::inverse;

  void forward(typename Base::CMap x, typename Base::Map y, float const s = 1.f) const final;
  void adjoint(typename Base::CMap y, typename Base::Map x, float const s = 1.f) const final;
  void inverse(typename Base::CMap y, typename Base::Map x, float const s = 1.f, float const b = 0.f) const final;

  void iforward(typename Base::CMap x, typename Base::Map y, float const s = 1.f) const final;
  void iadjoint(typename Base::CMap y, typename Base::Map x, float const s = 1.f) const final;

  virtual auto forward(InTensor const &x, float const s = 1.f) const -> OutTensor;
  virtual auto adjoint(OutTensor const &y, float const s = 1.f) const -> InTensor;
  virtual void forward(InTensor const &x, OutTensor &y, float const s = 1.f) const;
  virtual void adjoint(OutTensor const &y, InTensor &x, float const s = 1.f) const;

  virtual void forward(InCMap x, OutMap y, float const s = 1.f) const = 0;
  virtual void adjoint(OutCMap y, InMap x, float const s = 1.f) const = 0;
  virtual void inverse(OutCMap y, InMap x, float const s = 1.f, float const b = 0.f) const;
  virtual void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  virtual void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

protected:
  auto startForward(InCMap x, OutMap y, bool const ip) const -> Time;
  void finishForward(OutMap y, Time const start, bool const ip) const;

  auto startAdjoint(OutCMap y, InMap x, bool const ip) const -> Time;
  void finishAdjoint(InMap x, Time const start, bool const ip) const;

  auto startInverse(OutCMap y, InMap x) const -> Time;
  void finishInverse(InMap x, Time const start) const;
};

#define TOP_INHERIT(INRANK, OUTRANK)                                                                                           \
  using Parent = TOp<INRANK, OUTRANK>;                                                                                         \
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

#define TOP_DECLARE(SELF)                                                                                                      \
  using Ptr = std::shared_ptr<SELF>;                                                                                           \
  void forward(InCMap x, OutMap y, float const s = 1.f) const;                                                                 \
  void adjoint(OutCMap y, InMap x, float const s = 1.f) const;                                                                 \
  using Parent::forward;                                                                                                       \
  using Parent::adjoint;

} // namespace rl::TOps
