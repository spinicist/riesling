#include "ndft.hpp"

#include "op/loop.hpp"
#include "op/rank.hpp"

using namespace std::complex_literals;

namespace rl {

template <size_t NDim>
NDFTOp<NDim>::NDFTOp(
  Re3 const &tr, Index const nC, Sz<NDim> const shape, Re2 const &b, std::shared_ptr<TensorOperator<Cx, 3>> s)
  : Parent("NDFTOp", AddFront(shape, nC, b.dimension(0)), AddFront(LastN<2>(tr.dimensions()), nC))
  , sdc{s ? s : std::make_shared<TensorIdentity<Cx, 3>>(oshape)}
{
  static_assert(NDim < 4);
  if (tr.dimension(0) != NDim) { Log::Fail("Requested {}D NDFT but trajectory is {}D", NDim, tr.dimension(0)); }
  Log::Print<Log::Level::High>("NDFT Input Dims {} Output Dims {}", ishape, oshape);
  Log::Print<Log::Level::High>("Calculating cartesian co-ords");
  nSamp = tr.dimension(1);
  nTrace = tr.dimension(2);
  Index const M = nSamp * nTrace;
  N = Product(shape);
  scale = 1.f / std::sqrt(N);

  Eigen::Array<float, NDim, 1> trScale;
  using FMap = typename Eigen::Array<float, NDim, -1>::ConstAlignedMapType;
  FMap tm(tr.data(), NDim, M);
  for (Index ii = 0; ii < NDim; ii++) {
    trScale(ii) = shape[ii];
  }
  traj.resize(NDim, M);
  traj = ((tm + 0.5f).unaryExpr([](float const f) { return std::fmod(f, 1.f); }) - 0.5f).colwise() * trScale;

  Index const nB0 = b.dimension(0);
  Index const nB1 = b.dimension(1);
  basis = Eigen::ArrayXXf::ConstMapType(b.data(), nB0, nB1).replicate(1, 1 + nTrace / nB1).leftCols(nTrace).template cast<Cx>();
  xc.resize(NDim, N);
  Index       ind = 0;
  Index const si = shape[NDim - 1];
  for (int16_t ii = 0; ii < si; ii++) {
    float const fi = (float)(ii - si / 2) / si;
    if constexpr (NDim == 1) {
      xc(ind) = 2.f * M_PI * fi;
      ind++;
    } else {
      Index const sj = shape[NDim - 2];
      for (int16_t ij = 0; ij < sj; ij++) {
        float const fj = (float)(ij - sj / 2) / sj;
        if constexpr (NDim == 2) {
          xc.col(ind) = 2.f * M_PI * Eigen::Vector2f(fj, fi);
          ind++;
        } else {
          Index const sk = shape[NDim - 3];
          for (int16_t ik = 0; ik < sk; ik++) {
            float const fk = (float)(ik - sk / 2) / sk;
            xc.col(ind) = 2.f * M_PI * Eigen::Vector3f(fk, fj, fi);
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
  using RVec = typename Eigen::RowVector<float, NDim>;
  using CxMap = typename Eigen::Matrix<Cx, -1, 1>::AlignedMapType;
  using CxCMap = typename Eigen::Matrix<Cx, -1, -1>::ConstAlignedMapType;

  Index const nC = ishape[0];
  Index const nB = ishape[1];

  CxCMap xm(x.data(), nC, nB * N);
  CxMap  ym(y.data(), nC, nSamp * nTrace);
  auto task = [&](Index const itr) {
    Eigen::VectorXcf const b = basis.col(itr);
    for (Index isamp = 0; isamp < nSamp; isamp++) {
      Index const            ii = itr * nSamp + isamp;
      RVec const             f = -traj.col(ii).transpose();
      Eigen::VectorXf const  ph = f * xc;
      Eigen::RowVectorXcf const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
      Eigen::VectorXcf const samp = xm * (b * eph).reshaped(nB * N, 1);
      ym.col(ii) = samp * scale;
    }
  };
  Threads::For(task, nTrace, "NDFT Forward");
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

  using RVec = typename Eigen::RowVector<float, NDim>;
  using CxMap = typename Eigen::Matrix<Cx, -1, 1>::AlignedMapType;
  using CxCMap = typename Eigen::Matrix<Cx, -1, -1>::ConstAlignedMapType;

  Index const nC = ishape[0];
  Index const nB = ishape[1];

  CxCMap ym(y.data(), nC, nSamp * nTrace);
  CxMap  xm(x.data(), nC * nB, N);
  auto task = [&](Index ii) {
    RVec const             f = xc.col(ii).transpose();
    Eigen::MatrixXcf vox = Eigen::MatrixXcf::Zero(nC, nB);
    for (Index itr = 0; itr < nTrace; itr++) {
        Eigen::ArrayXf const  ph = f * traj.middleCols(itr * nSamp, nSamp);
        Eigen::VectorXcf const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
        Eigen::RowVectorXcf const b = basis.col(itr).transpose();
        vox += ym.middleCols(itr * nSamp, nSamp) * (eph * b);
    }
    xm.col(ii) = vox.reshaped(nC * nB, 1) * scale;
  };
  Threads::For(task, N, "NDFT Adjoint");
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
