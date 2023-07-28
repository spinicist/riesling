#include "ndft.hpp"

#include "op/loop.hpp"
#include "op/rank.hpp"

using namespace std::complex_literals;

namespace rl {

template <size_t NDim>
NDFTOp<NDim>::NDFTOp(
  Re3 const &traj, Index const nC, Sz<NDim> const shape, Re2 const &b, std::shared_ptr<TensorOperator<Cx, 3>> s)
  : Parent("NDFTOp", AddFront(shape, nC, b.dimension(0)), AddFront(LastN<2>(traj.dimensions()), nC))
  , traj{traj}
  , basis{b}
  , sdc{s ? s : std::make_shared<TensorIdentity<Cx, 3>>(oshape)}
{
  static_assert(NDim < 4);
  if (traj.dimension(0) != NDim) { Log::Fail("Requested {}D NDFT but trajectory is {}D", NDim, traj.dimension(0)); }
  Log::Print<Log::Level::High>("NDFT Input Dims {} Output Dims {}", ishape, oshape);
  Log::Print<Log::Level::High>("Calculating cartesian co-ords");
  scale = 1.f / std::sqrt(Product(ishape));
  xc.resize(NDim, Product(shape));
  xind.resize(Product(shape));
  Index ind = 0;
  for (int16_t ii = 0; ii < shape[NDim - 1]; ii++) {
    float const fi = (ii - shape[NDim - 1] / 2.f) / 2.f;
    if constexpr (NDim == 1) {
      xind[ind] = {ii};
      xc(ind) = 4.f * M_PI * fi;
      ind++;
    } else {
      for (int16_t ij = 0; ij < shape[NDim - 2]; ij++) {
        float const fj = (ij - shape[NDim - 2] / 2.f) / 2.f;
        if constexpr (NDim == 2) {
          xind[ind] = {ij, ii};
          xc.col(ind) = 4.f * M_PI * Eigen::Vector2f(fj, fi);
          ind++;
        } else {
          for (int16_t ik = 0; ik < shape[NDim - 3]; ik++) {
            float const fk = (ik - shape[NDim - 3] / 2.f) / 2.f;
            xind[ind] = {ik, ij, ii};
            xc.col(ind) = 4.f * M_PI * Eigen::Vector3f(fk, fj, fi);
            ind++;
          }
        }
      }
    }
  }
}

template <size_t NDim>
void NDFTOp<NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  auto       task = [&](Index is) {
    fmt::print(stderr, "is {}\n", is);
    Index const nC = ishape[0];
    Index const nB = ishape[1];

    float const pi = M_PI;
    for (Index ir = 0; ir < traj.dimension(1); ir++) {
      Re1         nc = traj.template chip<2>(is).template chip<1>(ir);
      Index const hi = ishape[2] / 2;
      for (Index ii = 0; ii < ishape[2]; ii++) {
        float const fi = (ii - hi) / (float)ishape[2];
        if constexpr (NDim == 1) {
          Cx const ph = std::exp(4.if * pi * (nc[0] * fi)) * scale;
          for (Index ib = 0; ib < nB; ib++) {
            for (Index ic = 0; ic < nC; ic++) {
              // fmt::print(stderr, "ic {} ib {} ii {} ir {} is {} ph {} x {} nc {} fi {} nc*fi {}\n", ic, ib, ii, ir, is, ph,
              //                  x(ic, ib, ii), nc[0], fi, nc[0] * fi);
              y(ic, ir, is) += x(ic, ib, ii) * ph;
            }
          }
        } else {
          float const hj = ishape[3] / 2;
          for (Index ij = 0; ij < ishape[3]; ij++) {
            float const fj = (ij - hj) / (float)ishape[3];
            if constexpr (NDim == 2) {
              Cx const ph = std::exp(4.if * pi * (nc[1] * fi + nc[0] * fj)) * scale;
              for (Index ib = 0; ib < nB; ib++) {
                for (Index ic = 0; ic < nC; ic++) {
                  y(ic, ir, is) += x(ic, ib, ii, ij) * ph;
                }
              }
            } else {
              float const hk = ishape[4] / 2;
              for (Index ik = 0; ik < ishape[4]; ik++) {
                float const fk = (ik - hk) / (float)ishape[4];
                Cx const    ph = std::exp(4.if * pi * (nc[2] * fi + nc[1] * fj + nc[0] * fk)) * scale;
                for (Index ib = 0; ib < nB; ib++) {
                  for (Index ic = 0; ic < nC; ic++) {
                    y(ic, ir, is) += x(ic, ib, ii, ij, ik) * ph;
                  }
                }
              }
            }
          }
        }
      }
    }
  };
  y.device(Threads::GlobalDevice()) = y.constant(0.f);
  Threads::For(task, traj.dimension(2), "NDFT Forward");
  this->finishForward(y, time);
}

template <size_t NDim>
void NDFTOp<NDim>::adjoint(OutCMap const &yy, InMap &x) const
{
  auto const time = this->startAdjoint(yy);
  OutTensor  sy;
  OutCMap    y(yy);
  if (sdc) {
    sy.resize(yy.dimensions());
    sy = sdc->adjoint(yy);
    new (&y) OutCMap(sy.data(), sy.dimensions());
  }

  using Map = typename Eigen::Matrix<float, NDim, -1>::ConstMapType;
  using RVec = typename Eigen::RowVector<float, NDim>;
  auto task = [&](Index ii) {
    Index const     nC = ishape[0];
    Index const     nB = ishape[1];
    RVec const      f = xc.col(ii).transpose();
    auto const      xi = xind[ii];
    Eigen::ArrayXf  ph(traj.dimension(1));
    Eigen::ArrayXcf eph(traj.dimension(1));
    for (Index is = 0; is < traj.dimension(2); is++) {
      Map m(&traj(0, 0, is), NDim, traj.dimension(1));
      ph = f * m;
      eph.real() = ph.array().cos() * scale;
      eph.imag() = ph.array().sin() * scale;
      for (Index ir = 0; ir < traj.dimension(1); ir++) {
        for (int16_t ib = 0; ib < nB; ib++) {
          for (int16_t ic = 0; ic < nC; ic++) {
            if constexpr (NDim == 1) {
              x(ic, ib, xi[0]) += y(ic, ir, is) * eph(ir);
            } else if constexpr (NDim == 2) {
              x(ic, ib, xi[0], xi[1]) += y(ic, ir, is) * eph(ir);
            } else if constexpr (NDim == 3) {
              x(ic, ib, xi[0], xi[1], xi[2]) += y(ic, ir, is) * eph(ir);
            }
          }
        }
      }
    }
  };

  x.device(Threads::GlobalDevice()) = x.constant(0.f);
  Threads::For(task, xind.size(), "NDFT Adjoint");
  this->finishAdjoint(x, time);
}

template struct NDFTOp<1>;
template struct NDFTOp<2>;
template struct NDFTOp<3>;

std::shared_ptr<TensorOperator<Cx, 5, 4>>
make_ndft(Re3 const &traj, Index const nC, Sz3 const matrix, Re2 const &basis, std::shared_ptr<TensorOperator<Cx, 3>> sdc)
{

  std::shared_ptr<TensorOperator<Cx, 5, 4>> ndft;
  if (traj.dimension(0) == 2) {
    Log::Print<Log::Level::Debug>("Creating 2D Multi-slice NDFT");
    auto ndft2 = std::make_shared<NDFTOp<2>>(traj, nC, FirstN<2>(matrix), basis, sdc);
    ndft = std::make_shared<LoopOp<NDFTOp<2>>>(ndft2, matrix[2]);
  } else {
    Log::Print<Log::Level::Debug>("Creating full 3D NDFT");
    ndft = std::make_shared<IncreaseOutputRank<NDFTOp<3>>>(std::make_shared<NDFTOp<3>>(traj, nC, matrix, basis, sdc));
  }
  return ndft;
}

} // namespace rl
