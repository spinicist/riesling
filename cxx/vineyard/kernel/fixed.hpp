#pragma once

#include "kernel.hpp"

namespace rl {

template <int ND, int W> struct KernelSizes
{
};

template <int W> struct KernelSizes<1, W>
{
  using Type = Eigen::Sizes<W>;
};

template <int W> struct KernelSizes<2, W>
{
  using Type = Eigen::Sizes<W, W>;
};

template <int W> struct KernelSizes<3, W>
{
  using Type = Eigen::Sizes<W, W, W>;
};

template <typename Scalar, int ND, int W> struct FixedKernel : Kernel<Scalar, ND>
{
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<ND, W>::Type>;
  using Point = Eigen::Matrix<float, ND, 1>;

  OneD centers;

  FixedKernel()
  {
    Eigen::TensorFixedSize<float, Eigen::Sizes<W>> pos;
    for (int ii = 0; ii < W; ii++) {
      centers(ii) = ii + 0.5f - (W / 2.f);
    }
  }

  virtual auto paddedWidth() const -> int final { return W; };
  virtual auto at(Point const p) const -> Eigen::Tensor<float, ND> final { return this->operator()(p); }
  virtual auto operator()(Point const p) const -> Tensor = 0;

  inline void spreadInner(Eigen::Tensor<Scalar, 1> const &y,
                          Eigen::Tensor<Scalar, 1> const &b,
                          float const                     kval,
                          Sz<ND + 2>                      xInd,
                          Eigen::Tensor<Scalar, ND + 2>  &x) const
  {
    int const nC = x.dimension(0);
    int const nB = b.size();
    for (int ib = 0; ib < nB; ib++) {
      xInd[1] = ib;
      Scalar const bval = kval * b(ib);
      for (int ic = 0; ic < nC; ic++) {
        xInd[0] = ic;
        x(xInd) += y(ic) * bval;
      }
    }
  }

  template <int D>
  inline void spreadSpatial(Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 1> const &b,
                            Sz<ND>                          kInd,
                            Tensor const                   &k,
                            Sz<ND> const                    off,
                            Sz<ND + 2>                      xInd,
                            Eigen::Tensor<Scalar, ND + 2>  &x) const
  {
    for (int ii = 0; ii < W; ii++) {
      kInd[D] = ii;
      xInd[D + 2] = ii + off[D];
      if constexpr (D == 0) {
        float const kval = k(kInd);
        spreadInner(y, b, kval, xInd, x);
      } else {
        spreadSpatial<D - 1>(y, b, kInd, k, off, xInd, x);
      }
    }
  }

  inline void spread(std::array<int16_t, ND> const   c,
                     Point const                     p,
                     Sz<ND> const                    minCorner,
                     Eigen::Tensor<Scalar, 1> const &b,
                     Eigen::Tensor<Scalar, 1> const &y,
                     Eigen::Tensor<Scalar, ND + 2>  &x) const final
  {
    Tensor const k = this->operator()(p);
    Sz<ND>       off;
    for (int id = 0; id < ND; id++) {
      off[id] = c[id] - minCorner[id] - W / 2;
    }
    Sz<ND>     kInd;
    Sz<ND + 2> xInd;
    spreadSpatial<ND - 1>(y, b, kInd, k, off, xInd, x);
  }

  inline void gatherInner(Sz<ND + 2>                                  xInd,
                          Eigen::Tensor<Scalar, ND + 2> const        &x,
                          Eigen::Tensor<Scalar, 1> const             &b,
                          float const                                 kval,
                          Eigen::TensorMap<Eigen::Tensor<Scalar, 1>> &y) const
  {
    int const nC = x.dimension(0);
    int const nB = b.size();
    for (int ib = 0; ib < nB; ib++) {
      xInd[1] = ib;
      Scalar const bval = kval * b(ib);
      for (int ic = 0; ic < nC; ic++) {
        xInd[0] = ic;
        y(ic) += x(xInd) * bval;
      }
    }
  }

  template <int D>
  inline void gatherSpatial(Sz<ND> const                                off,
                            Sz<ND + 2>                                  xInd,
                            Eigen::Tensor<Scalar, ND + 2> const        &x,
                            Eigen::Tensor<Scalar, 1> const             &b,
                            Sz<ND>                                      kInd,
                            Tensor const                               &k,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>> &y) const
  {
    for (int ii = 0; ii < W; ii++) {
      kInd[D] = ii;
      xInd[D + 2] = ii + off[D];
      if constexpr (D == 0) {
        float const kval = k(kInd);
        gatherInner(xInd, x, b, kval, y);
      } else {
        gatherSpatial<D - 1>(off, xInd, x, b, kInd, k, y);
      }
    }
  }

  void gather(std::array<int16_t, ND> const                                c,
              Point const                                                  p,
              Sz<ND> const                                                 minCorner,
              Eigen::Tensor<Scalar, 1> const                              &b,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    Tensor const k = this->operator()(p);
    Sz<ND>       off;
    for (int id = 0; id < ND; id++) {
      off[id] = c[id] - minCorner[id] - W / 2;
    }
    Sz<ND>     kInd;
    Sz<ND + 2> xInd;
    gatherSpatial<ND - 1>(off, xInd, x, b, kInd, k, y);
  }
};

} // namespace rl
