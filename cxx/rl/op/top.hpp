#pragma once

#include "ops.hpp"
#include "../tensors.hpp"

namespace rl::TOps {

template <typename Scalar_, int InRank_, int OutRank_ = InRank_> struct TOp : Ops::Op<Scalar_>
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

  void forward(typename Base::CMap const &x, typename Base::Map &y) const final;
  void adjoint(typename Base::CMap const &y, typename Base::Map &x) const final;

  void iforward(typename Base::CMap const &x, typename Base::Map &y) const final;
  void iadjoint(typename Base::CMap const &y, typename Base::Map &x) const final;

  virtual auto forward(InTensor const &x) const -> OutTensor;
  virtual auto adjoint(OutTensor const &y) const -> InTensor;
  virtual void forward(InTensor const &x, OutTensor &y) const;
  virtual void adjoint(OutTensor const &y, InTensor &x) const;

  virtual void forward(InCMap const &x, OutMap &y) const = 0;
  virtual void adjoint(OutCMap const &y, InMap &x) const = 0;
  virtual void iforward(InCMap const &x, OutMap &y) const;
  virtual void iadjoint(OutCMap const &y, InMap &x) const;

protected:
  auto startForward(InCMap const &x, OutMap const &y, bool const ip) const -> Time;
  void finishForward(OutMap const &y, Time const start, bool const ip) const;

  auto startAdjoint(OutCMap const &y, InMap const &x, bool const ip) const -> Time;
  void finishAdjoint(InMap const &x, Time const start, bool const ip) const;
};

#define TOP_INHERIT(SCALAR, INRANK, OUTRANK)                                                                                   \
  using Parent = TOp<SCALAR, INRANK, OUTRANK>;                                                                                 \
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

#define TOP_DECLARE(SELF)                                                                                                      \
  using Ptr = std::shared_ptr<SELF>;                                                                                           \
  void forward(InCMap const &x, OutMap &y) const;                                                                              \
  void adjoint(OutCMap const &y, InMap &x) const;                                                                              \
  using Parent::forward;                                                                                                       \
  using Parent::adjoint;

template <typename Scalar_, int Rank> struct Identity : TOp<Scalar_, Rank, Rank>
{
  TOP_INHERIT(Scalar_, Rank, Rank)
  Identity(Sz<Rank> dims);

  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;

  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;
};

} // namespace rl::TOps
