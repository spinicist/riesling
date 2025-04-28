#include "ndft.hpp"

#include "loop.hpp"
#include "pad.hpp"
#include "reshape.hpp"
#include "compose.hpp"
#include "multiplex.hpp"
#include "top-impl.hpp"

using namespace std::complex_literals;

namespace rl::TOps {

template <int NDim> NDFT<NDim>::NDFT(Sz<NDim> const shape, Re3 const &tr, Index const nC, Basis::CPtr b)
  : Parent("NDFT", AddBack(shape, nC, b ? b->nB() : 1), AddFront(LastN<2>(tr.dimensions()), nC))
  , basis{b}
{
  static_assert(NDim < 4);
  if (tr.dimension(0) != NDim) { throw Log::Failure("NDFT", "Requested {}D but trajectory is {}D", NDim, tr.dimension(0)); }
  Log::Debug("NDFT", "ishape {} oshape {}", ishape, oshape);
  Log::Debug("NDFT", "Calculating cartesian co-ords");
  nSamp = tr.dimension(1);
  nTrace = tr.dimension(2);
  N = Product(shape);
  scale = 1.f / std::sqrt(N);

  // Re1 trScale(NDim);
  // for (Index ii = 0; ii < NDim; ii++) {
  //   trScale(ii) = shape[ii];
  // }
  // traj = ((tr + 0.5f).unaryExpr([](float const f) { return std::fmod(f, 1.f); }) - 0.5f) *
  //        trScale.reshape(Sz3{NDim, 1, 1}).broadcast(Sz3{1, nSamp, nTrace});
  traj = tr;

  xc.resize(NDim, N);
  Index       ind = 0;
  Index const s0 = shape[NDim - 1];
  for (int16_t i0 = 0; i0 < s0; i0++) {
    float const f0 = (float)(i0 - s0 / 2) / s0;
    if constexpr (NDim == 1) {
      xc(0, ind) = 2.f * M_PI * f0;
      ind++;
    } else {
      Index const s1 = shape[NDim - 2];
      for (int16_t i1 = 0; i1 < s1; i1++) {
        float const f1 = (float)(i1 - s1 / 2) / s1;
        if constexpr (NDim == 2) {
          xc(0, ind) = 2.f * M_PI * f1;
          xc(1, ind) = 2.f * M_PI * f0;
          ind++;
        } else {
          Index const s2 = shape[NDim - 3];
          for (int16_t i2 = 0; i2 < s2; i2++) {
            float const f2 = (float)(i2 - s2 / 2) / s2;
            xc(0, ind) = 2.f * M_PI * f2;
            xc(1, ind) = 2.f * M_PI * f1;
            xc(2, ind) = 2.f * M_PI * f0;
            ind++;
          }
        }
      }
    }
  }
}

template <int NDim> auto NDFT<NDim>::Make(Sz<NDim> const matrix, Re3 const &traj, Index const nC, Basis::CPtr basis)
  -> std::shared_ptr<NDFT<NDim>>
{
  return std::make_shared<NDFT<NDim>>(matrix, traj, nC, basis);
}

template <int NDim> void NDFT<NDim>::addOffResonance(Eigen::Tensor<float, NDim> const &f0map, float const t0, float const)
{
  TOps::Pad<float, NDim> pad(f0map.dimensions(), LastN<NDim>(ishape));
  Δf.resize(N);
  assert(N == pad.rows());
  typename TOps::Pad<float, NDim>::OutMap fm(Δf.data(), pad.oshape);
  pad.forward(f0map, fm);
  t.resize(nSamp);
  t[0] = t0;
  for (Index ii = 1; ii < nSamp; ii++) {
    t[ii] = t[ii - 1] + t0;
  }
  Log::Print("NDFT", "Off-resonance correction. f0 range is {} to {} Hz", Minimum(Δf), Maximum(Δf));
}

template <int NDim> void NDFT<NDim>::forward(InCMap const x, OutMap y) const
{
  auto const  time = this->startForward(x, y, false);
  Index const nC = ishape[NDim - 2];
  Index const nB = ishape[NDim - 1];
  auto const  xr = x.reshape(Sz3{N, nC, nB});

  auto task = [&](Index const trlo, Index const trhi) {
    for (Index itr = trlo; itr < trhi; itr++) {
      for (Index isamp = 0; isamp < nSamp; isamp++) {
        Re1 ph = -traj.template chip<2>(itr).template chip<1>(isamp).broadcast(Sz2{1, N}).contract(
          xc, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
        if (Δf.size()) { ph += Δf * t[isamp] * 2.f * (float)M_PI; }
        Cx1 const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
        Cx2 const samp = xr.contract(eph, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
        if (basis) {
          Cx1 const b = basis->B.chip<2>(itr % basis->nSample()).template chip<1>(isamp % basis->nTrace());
          y.template chip<2>(itr).template chip<1>(isamp) =
            samp.contract(b, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()) * Cx(scale);
        } else {
          y.template chip<2>(itr).template chip<1>(isamp) = samp * Cx(scale);
        }
      }
    }
  };
  Threads::ChunkFor(task, nTrace);
  this->finishForward(y, time, false);
}

template <int NDim> void NDFT<NDim>::adjoint(OutCMap const yy, InMap x) const
{
  auto const                             time = this->startAdjoint(yy, x, false);
  OutTensor                              sy;
  OutCMap                                y(yy);
  Index const                            nC = ishape[InRank - 2];
  Index const                            nB = ishape[InRank - 1];
  Eigen::TensorMap<Eigen::Tensor<Cx, 3>> xm(x.data(), N, nC, nB);

  auto task = [&](Index const ilo, Index const ihi) {
    for (Index ii = ilo; ii < ihi; ii++) {
      Re1 const xf = xc.chip<1>(ii);
      Cx2       vox(nC, nB);
      vox.setZero();
      for (Index itr = 0; itr < nTrace; itr++) {
        Re1 ph = traj.chip<2>(itr).contract(xf, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
        if (Δf.size()) { ph -= Δf(ii) * t * 2.f * (float)M_PI; }
        Cx1 const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
        if (basis) {
          for (Index ib = 0; ib < nB; ib++) {
            Cx1 const b = basis->B.template chip<2>(itr % basis->nTrace())
                            .template chip<0>(ib)
                            .conjugate()
                            .broadcast(Sz1{nSamp / basis->nSample()});
            vox.chip<1>(ib) += y.template chip<2>(itr).contract(eph * b, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>());
          }
        } else {
          vox.chip<1>(0) += y.template chip<2>(itr).contract(eph, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>());
        }
      }
      xm.chip<0>(ii) = vox * Cx(scale);
    }
  };
  Threads::ChunkFor(task, N);
  this->finishAdjoint(x, time, false);
}

template struct NDFT<1>;
template struct NDFT<2>;
template struct NDFT<3>;

auto NDFTAll(Sz3 const shape, Re3 const &tr, Index const nC, Index const nSlab, Index const nTime, Basis::CPtr b)
  -> TOps::TOp<Cx, 6, 5>::Ptr
{
  auto ndft = TOps::NDFT<3>::Make(shape, tr, nC, b);
  if (nSlab == 1) {
    auto reshape = TOps::MakeReshapeOutput(ndft, AddBack(ndft->oshape, 1));
    auto timeLoop = TOps::MakeLoop(reshape, nTime);
    return timeLoop;
  } else {
    auto loop = TOps::MakeLoop(ndft, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(ndft->ishape, nSlab);
    auto compose1 = TOps::MakeCompose(slabToVol, loop);
    auto timeLoop = TOps::MakeLoop(compose1, nTime);
    return timeLoop;
  }
}

} // namespace rl::TOps
