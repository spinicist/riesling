#include "ndft.hpp"

#include "../log/log.hpp"
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

template <int NDim> auto NDFT<NDim>::Make(Sz<NDim> const matrix, Re3 const &traj, Index const nC, Basis::CPtr basis) -> Ptr
{
  return std::make_shared<NDFT<NDim>>(matrix, traj, nC, basis);
}

template <int NDim> void NDFT<NDim>::addOffResonance(ReN<NDim> const &f0map, float const t0, float const)
{
  if (f0map.dimensions() != FirstN<NDim>(ishape)) {
    throw Log::Failure("NDFT", "Off-resonance map dimensions were {} should be {}", f0map.dimensions(), FirstN<NDim>(ishape));
  }
  Δf = f0map.reshape(Sz1{N});
  t.resize(nSamp);
  t[0] = t0;
  for (Index ii = 1; ii < nSamp; ii++) {
    t[ii] = t[ii - 1] + t0;
  }
  Log::Print("NDFT", "Off-resonance correction. f0 range is {} to {} Hz", Minimum(Δf), Maximum(Δf));
}

template <int NDim> void NDFT<NDim>::forward(InCMap x, OutMap y) const
{
  auto const  time = this->startForward(x, y, false);
  Index const nC = ishape[InRank - 2];

  if (basis) {
    Index const nB = ishape[InRank - 1];
    auto const  xr = x.reshape(Sz3{N, nC, nB});
    auto        task = [&](Index const trlo, Index const trhi) {
      for (Index itr = trlo; itr < trhi; itr++) {
        for (Index isamp = 0; isamp < nSamp; isamp++) {
          Re1 ph = -traj.template chip<2>(itr).template chip<1>(isamp).broadcast(Sz2{1, N}).contract(
            xc, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
          if (Δf.size()) { ph += Δf * t[isamp] * 2.f * (float)M_PI; }
          Cx1 const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
          Cx2 const samp = xr.contract(eph, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
          Cx1 const b = basis->B.chip<2>(itr % basis->nSample()).template chip<1>(isamp % basis->nTrace());
          y.template chip<2>(itr).template chip<1>(isamp) =
            samp.contract(b, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()) * Cx(scale);
        }
      }
    };
    Threads::ChunkFor(task, nTrace);
  } else {
    auto task = [&](Index const trlo, Index const trhi) {
      for (Index itr = trlo; itr < trhi; itr++) {
        for (Index isamp = 0; isamp < nSamp; isamp++) {
          for (Index ic = 0; ic < nC; ic++) {
            y(ic, isamp, itr) = 0.f;
          }
          Index const s0 = ishape[NDim - 1];
          for (Index i0 = 0; i0 < s0; i0++) {
            float const p0 = -traj(0, isamp, itr) * (2.f * M_PI * (i0 - s0 / 2)) / s0;
            if constexpr (NDim == 1) {
              Cx const f(std::cos(p0), std::sin(p0));
              for (Index ic = 0; ic < nC; ic++) {
                y(ic, isamp, itr) += f * x(i0, ic, 0);
              }
            } else {
              Index const s1 = ishape[NDim - 2];
              for (Index i1 = 0; i1 < s1; i1++) {
                float const p1 = p0 - traj(1, isamp, itr) * (2.f * M_PI * (i1 - s1 / 2)) / s1;
                if constexpr (NDim == 2) {
                  Cx const f(std::cos(p1), std::sin(p1));
                  for (Index ic = 0; ic < nC; ic++) {
                    y(ic, isamp, itr) += f * x(i0, i1, ic, 0);
                  }
                } else {
                  Index const s2 = ishape[NDim - 3];
                  for (Index i2 = 0; i2 < s2; i2++) {
                    float const p2 = p1 - traj(2, isamp, itr) * (2.f * M_PI * (i2 - s2 / 2)) / s2;
                    Cx const    f(std::cos(p2), std::sin(p2));
                    for (Index ic = 0; ic < nC; ic++) {
                      y(ic, isamp, itr) += f * x(i0, i1, i2, ic, 0);
                    }
                  }
                }
              }
            }
          }
        }
      }
    };
    Threads::ChunkFor(task, nTrace);
  }
  this->finishForward(y, time, false);
}

template <int NDim> void NDFT<NDim>::iforward(InCMap x, OutMap y) const
{
  auto const  time = this->startForward(x, y, false);
  Index const nC = ishape[InRank - 2];

  if (basis) {
    Index const nB = ishape[InRank - 1];
    auto const  xr = x.reshape(Sz3{N, nC, nB});
    auto        task = [&](Index const trlo, Index const trhi) {
      for (Index itr = trlo; itr < trhi; itr++) {
        for (Index isamp = 0; isamp < nSamp; isamp++) {
          Re1 ph = -traj.template chip<2>(itr).template chip<1>(isamp).broadcast(Sz2{1, N}).contract(
            xc, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
          if (Δf.size()) { ph += Δf * t[isamp] * 2.f * (float)M_PI; }
          Cx1 const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
          Cx2 const samp = xr.contract(eph, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
          Cx1 const b = basis->B.chip<2>(itr % basis->nSample()).template chip<1>(isamp % basis->nTrace());
          y.template chip<2>(itr).template chip<1>(isamp) +=
            samp.contract(b, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()) * Cx(scale);
        }
      }
    };
    Threads::ChunkFor(task, nTrace);
  } else {
    auto task = [&](Index const trlo, Index const trhi) {
      for (Index itr = trlo; itr < trhi; itr++) {
        for (Index isamp = 0; isamp < nSamp; isamp++) {
          Index const s0 = ishape[NDim - 1];
          for (Index i0 = 0; i0 < s0; i0++) {
            float const p0 = -traj(0, isamp, itr) * (2.f * M_PI * (i0 - s0 / 2)) / s0;
            if constexpr (NDim == 1) {
              Cx const f(std::cos(p0), std::sin(p0));
              for (Index ic = 0; ic < nC; ic++) {
                y(ic, isamp, itr) += f * x(i0, ic, 0);
              }
            } else {
              Index const s1 = ishape[NDim - 2];
              for (Index i1 = 0; i1 < s1; i1++) {
                float const p1 = p0 - traj(1, isamp, itr) * (2.f * M_PI * (i1 - s1 / 2)) / s1;
                if constexpr (NDim == 2) {
                  Cx const f(std::cos(p1), std::sin(p1));
                  for (Index ic = 0; ic < nC; ic++) {
                    y(ic, isamp, itr) += f * x(i0, i1, ic, 0);
                  }
                } else {
                  Index const s2 = ishape[NDim - 3];
                  for (Index i2 = 0; i2 < s2; i2++) {
                    float const p2 = p1 - traj(2, isamp, itr) * (2.f * M_PI * (i2 - s2 / 2)) / s2;
                    Cx const    f(std::cos(p2), std::sin(p2));
                    for (Index ic = 0; ic < nC; ic++) {
                      y(ic, isamp, itr) += f * x(i0, i1, i2, ic, 0);
                    }
                  }
                }
              }
            }
          }
        }
      }
    };
    Threads::ChunkFor(task, nTrace);
  }
  this->finishForward(y, time, false);
}

template <int NDim> void NDFT<NDim>::adjoint(OutCMap yy, InMap x) const
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

template <int NDim> void NDFT<NDim>::iadjoint(OutCMap yy, InMap x) const
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
      xm.chip<0>(ii) += vox * Cx(scale);
    }
  };
  Threads::ChunkFor(task, N);
  this->finishAdjoint(x, time, false);
}

template <int NDim> auto NDFT<NDim>::M(float const λ, Index const nS, Index const nT) const -> TOps::TOp<5, 5>::Ptr
{
  Log::Print("NDFT", "Calculating preconditioner λ {}", λ);
  Cx3 ones(this->oshape);
  ones.setConstant(1.f);
  auto xcor = this->adjoint(ones);
  xcor.device(Threads::TensorDevice()) = xcor * xcor.conjugate();
  Re3         weights = (1.f + λ) / (this->forward(xcor).abs() + λ);
  float const norm = Norm<true>(weights);
  if (!std::isfinite(norm)) {
    throw Log::Failure("NDFT", "Pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("NDFT", "Pre-conditioner finished, norm {} min {} max {}", norm, Minimum(weights), Maximum(weights));
  }

  Sz5 const shape = AddBack(this->oshape, nS, nT);
  return std::make_shared<TOps::TensorScale<Cx, 5, 0, 2>>(shape, weights.cast<Cx>());
}

template struct NDFT<1>;
template struct NDFT<2>;
template struct NDFT<3>;

} // namespace rl::TOps
