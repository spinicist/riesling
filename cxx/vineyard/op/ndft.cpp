#include "ndft.hpp"

#include "op/loop.hpp"
#include "op/pad.hpp"
#include "op/rank.hpp"

using namespace std::complex_literals;

namespace rl {

template <int NDim>
NDFTOp<NDim>::NDFTOp(
  Re3 const &tr, Index const nC, Sz<NDim> const shape, Basis<Cx> const &b, std::shared_ptr<TensorOperator<Cx, 3>> s)
  : Parent("NDFTOp", AddFront(shape, nC, b.dimension(0)), AddFront(LastN<2>(tr.dimensions()), nC))
  , basis{b}
  , sdc{s ? s : std::make_shared<TensorIdentity<Cx, 3>>(oshape)}
{
  static_assert(NDim < 4);
  if (tr.dimension(0) != NDim) { Log::Fail("Requested {}D NDFT but trajectory is {}D", NDim, tr.dimension(0)); }
  Log::Debug("NDFT Input Dims {} Output Dims {}", ishape, oshape);
  Log::Debug("Calculating cartesian co-ords");
  nSamp = tr.dimension(1);
  nTrace = tr.dimension(2);
  N = Product(shape);
  scale = 1.f / std::sqrt(N);

  Re1 trScale(NDim);
  for (Index ii = 0; ii < NDim; ii++) {
    trScale(ii) = shape[ii];
  }
  traj = ((tr + 0.5f).unaryExpr([](float const f) { return std::fmod(f, 1.f); }) - 0.5f) *
         trScale.reshape(Sz3{NDim, 1, 1}).broadcast(Sz3{1, nSamp, nTrace});

  xc.resize(NDim, N);
  Index       ind = 0;
  Index const si = shape[NDim - 1];
  for (int16_t ii = 0; ii < si; ii++) {
    float const fi = (float)(ii - si / 2) / si;
    if constexpr (NDim == 1) {
      xc(0, ind) = 2.f * M_PI * fi;
      ind++;
    } else {
      Index const sj = shape[NDim - 2];
      for (int16_t ij = 0; ij < sj; ij++) {
        float const fj = (float)(ij - sj / 2) / sj;
        if constexpr (NDim == 2) {
          xc(0, ind) = 2.f * M_PI * fi;
          xc(1, ind) = 2.f * M_PI * fj;
          ind++;
        } else {
          Index const sk = shape[NDim - 3];
          for (int16_t ik = 0; ik < sk; ik++) {
            float const fk = (float)(ik - sk / 2) / sk;
            xc(0, ind) = 2.f * M_PI * fi;
            xc(1, ind) = 2.f * M_PI * fj;
            xc(2, ind) = 2.f * M_PI * fk;
            ind++;
          }
        }
      }
    }
  }
}

template <int NDim>
void NDFTOp<NDim>::addOffResonance(Eigen::Tensor<float, NDim> const &f0map, float const t0, float const tSamp)
{
  PadOp<float, NDim, NDim> pad(f0map.dimensions(), LastN<NDim>(ishape));
  Δf.resize(N);
  assert(N == pad.rows());
  typename PadOp<float, NDim, NDim>::OutMap fm(Δf.data(), pad.oshape);
  pad.forward(f0map, fm);
  t.resize(nSamp);
  t[0] = t0;
  for (Index ii = 1; ii < nSamp; ii++) {
    t[ii] = t[ii - 1] + t0;
  }
  Log::Print("Off-resonance correction. f0 range is {} to {} Hz", Minimum(Δf), Maximum(Δf));
}

template <int NDim>
void NDFTOp<NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const  time = this->startForward(x);
  Index const nC = ishape[0];
  Index const nV = ishape[1];
  auto const  xr = x.reshape(Sz3{nC, nV, N});

  auto task = [&](Index const itr) {
    for (Index isamp = 0; isamp < nSamp; isamp++) {
      Cx1 const b = basis.chip<2>(itr % basis.dimension(2)).template chip<1>(isamp % basis.dimension(1));
      Re1       ph = -traj.template chip<2>(itr).template chip<1>(isamp).broadcast(Sz2{1, N}).contract(
        xc, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
      if (Δf.size()) { ph += Δf * t[isamp] * 2.f * (float)M_PI; }
      Cx1 const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
      Cx1 const samp = xr.contract(eph, Eigen::IndexPairList<Eigen::type2indexpair<2, 0>>())
                         .contract(b, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>());
      y.template chip<2>(itr).template chip<1>(isamp) = samp * Cx(scale);
    }
  };
  Threads::For(task, nTrace, "NDFT Forward");
  this->finishForward(y, time);
}

template <int NDim>
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
  Index const                            nC = ishape[0];
  Index const                            nV = ishape[1];
  Eigen::TensorMap<Eigen::Tensor<Cx, 3>> xm(x.data(), nC, nV, N);

  auto task = [&](Index ii) {
    Re1 const xf = xc.chip<1>(ii);
    Cx2       vox(nC, nV);
    vox.setZero();
    for (Index itr = 0; itr < nTrace; itr++) {
      Re1 ph = traj.chip<2>(itr).contract(xf, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
      if (Δf.size()) { ph -= Δf(ii) * t * 2.f * (float)M_PI; }
      Cx1 const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
      for (Index iv = 0; iv < nV; iv++) {
        Cx1 const b = basis.template chip<2>(itr % basis.dimension(2))
                        .template chip<0>(iv)
                        .conjugate()
                        .broadcast(Sz1{nSamp / basis.dimension(1)});
        vox.chip<1>(iv) += y.template chip<2>(itr).contract(eph * b, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>());
      }
    }
    xm.chip<2>(ii) = vox * Cx(scale);
  };
  Threads::For(task, N, "NDFT Adjoint");
  this->finishAdjoint(x, time);
}

template struct NDFTOp<1>;
template struct NDFTOp<2>;
template struct NDFTOp<3>;

std::shared_ptr<TensorOperator<Cx, 5, 4>>
make_ndft(Re3 const &traj, Index const nC, Sz3 const matrix, Basis<Cx> const &basis, std::shared_ptr<TensorOperator<Cx, 3>> sdc)
{
  std::shared_ptr<TensorOperator<Cx, 5, 4>> ndft;
  if (traj.dimension(0) == 2) {
    Log::Debug("Creating 2D Multi-slice NDFT");
    auto ndft2 = std::make_shared<NDFTOp<2>>(traj, nC, FirstN<2>(matrix), basis, sdc);
    ndft = std::make_shared<LoopOp<NDFTOp<2>>>(ndft2, matrix[2]);
  } else {
    Log::Debug("Creating full 3D NDFT");
    auto ndft3 = std::make_shared<NDFTOp<3>>(traj, nC, matrix, basis, sdc);
    ndft = std::make_shared<IncreaseOutputRank<NDFTOp<3>>>(ndft3);
  }
  return ndft;
}

} // namespace rl
