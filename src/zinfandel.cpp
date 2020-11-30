#include "parse_args.h"
#include "threads.h"
#include "zinfandel.h"
#include <Eigen/SVD>
#include <complex>

using namespace std::complex_literals;

Eigen::MatrixXcd GrabSources(
    Cx3 const &ks,
    float const scale,
    long const spoke,
    long const st_read,
    long const n_src,
    long const n_cal)
{
  long const n_coil = ks.dimension(0);
  long const n_spoke = 5;
  long const lo_spoke = std::max(0L, spoke - n_spoke / 2);
  long const hi_spoke = std::min(ks.dimension(2), spoke + n_spoke / 2);
  long const n2_spoke = hi_spoke - lo_spoke;
  Eigen::MatrixXcd S(n_coil * n_src, n_cal * n2_spoke);
  S.setZero();
  for (long ispoke = lo_spoke; ispoke < hi_spoke; ispoke++) {
    long const st_spoke = (ispoke - lo_spoke) * n_cal;
    for (long ical = 0; ical < n_cal; ical++) {
      long const ind_cal = st_spoke + ical;
      for (long isrc = 0; isrc < n_src; isrc++) {
        for (long icoil = 0; icoil < n_coil; icoil++) {
          long const st_src = icoil * n_src;
          S(st_src + isrc, ind_cal) = ks(icoil, st_read + ical + isrc, ispoke) / scale;
        }
      }
    }
  }
  return S;
}

Eigen::MatrixXcd GrabSourcesOne(
    Cx3 const &ks,
    float const scale,
    long const spoke,
    long const st_read,
    long const n_src,
    long const n_cal)
{
  long const n_coil = ks.dimension(0);
  Eigen::MatrixXcd S(n_coil * n_src, n_cal);
  for (long ical = 0; ical < n_cal; ical++) {
    for (long isrc = 0; isrc < n_src; isrc++) {
      for (long icoil = 0; icoil < n_coil; icoil++) {
        long const st_src = icoil * n_src;
        S(st_src + isrc, ical) = ks(icoil, st_read + ical + isrc, spoke) / scale;
      }
    }
  }
  return S;
}

Eigen::MatrixXcd GrabTargets(
    Cx3 const &ks,
    float const scale,
    long const spoke,
    long const st_read,
    long const n_tgt,
    long const n_cal)
{
  long const n_coil = ks.dimension(0);
  long const n_spoke = 5;
  long const lo_spoke = std::max(0L, spoke - n_spoke / 2);
  long const hi_spoke = std::min(ks.dimension(2), spoke + n_spoke / 2);
  long const n2_spoke = hi_spoke - lo_spoke;

  Eigen::MatrixXcd T(n_coil * n_tgt, n_cal * n2_spoke);
  for (long ispoke = lo_spoke; ispoke < hi_spoke; ispoke++) {
    long const st_spoke = (ispoke - lo_spoke) * n_cal;
    for (long ical = 0; ical < n_cal; ical++) {
      for (long icoil = 0; icoil < n_coil; icoil++) {
        long const st_tgt = icoil * n_tgt;
        for (long itgt = 0; itgt < n_tgt; itgt++) {
          T(st_tgt + itgt, st_spoke + ical) = ks(icoil, st_read + ical + itgt, ispoke) / scale;
        }
      }
    }
  }
  return T;
}

void FillIterative(
    Eigen::VectorXcd const &S,
    Eigen::MatrixXcd const &W,
    float const scale,
    long const spoke,
    long const n_tgt,
    Cx3 &ks)
{
  long const n_coil = ks.dimension(0);
  long const n_rows = S.rows();
  long const n_src = n_rows / n_coil;
  Eigen::VectorXcd Si = S;
  Eigen::VectorXcd T(n_coil);
  for (long ii = 0; ii < n_tgt; ii++) {
    long const ir = n_tgt - 1 - ii;
    double const w0 = (ir == 0) ? 0.5 : 1.;
    T = W * Si;
    for (long icoil = 0; icoil < n_coil; icoil++) {
      ks(icoil, ir, spoke) = T(icoil) * (w0 * scale);
    }
    Si.tail(n_rows - 1) = Si.head(n_rows - 1).eval(); // Shift everything along, ensure temporary
    Eigen::VectorXcd check = Si(Eigen::seqN(0, n_coil, n_src));
    Si(Eigen::seqN(0, n_coil, n_src)) = T; // Copy in new values from T
  }
}

void FillSimultaneous(
    Eigen::VectorXcd const &S,
    Eigen::MatrixXcd const &W,
    float const scale,
    long const spoke,
    Cx3 &ks)
{
  long const n_coil = ks.dimension(0);
  long const n_tgt = W.rows() / n_coil;
  auto const T = W * S;
  for (long itgt = 0; itgt < n_tgt; itgt++) {
    double const w0 = (itgt == 0) ? 0.5 : 1.;
    for (long icoil = 0; icoil < n_coil; icoil++) {
      long const st_tgt = icoil * n_tgt;
      ks(icoil, itgt, spoke) = T(st_tgt + itgt) * (w0 * scale);
    }
  }
}

Eigen::MatrixXcd
CalcWeights(Eigen::MatrixXcd const &src, Eigen::MatrixXcd const tgt, float const lambda)
{
  if (lambda > 0.f) {
    auto const reg = lambda * src.norm() * Eigen::MatrixXcd::Identity(src.rows(), src.rows());
    auto const rhs = src * src.adjoint() + reg;
    Eigen::MatrixXcd pinv = rhs.completeOrthogonalDecomposition().pseudoInverse();
    return tgt * src.adjoint() * pinv;
  } else {
    Eigen::MatrixXcd const pinv = src.completeOrthogonalDecomposition().pseudoInverse();
    return tgt * pinv;
  }
}

void zinfandel(
    ZMode const mode,
    long const gap_sz,
    long const n_src,
    long const n_cal_in,
    float const lambda,
    Cx3 &ks,
    Log &log)
{
  bool is_simul = (mode == ZMode::Z1_simul);

  long const n_read = ks.dimension(1);

  long const n_tgt = is_simul ? gap_sz : 1;
  long const n_cal = n_cal_in < 1 ? n_read - (gap_sz + n_tgt + n_src - 1) : n_cal_in;

  log.info(
      FMT_STRING("ZINFANDEL {} Sources {} Calibration Region Size {} Gap {} Targets {}"),
      is_simul ? "Simultaneous" : "Iterative",
      n_src,
      n_cal,
      gap_sz,
      n_tgt);

  auto spoke_task = [&](long const spoke_lo, long const spoke_hi) {
    float rmse_accum = 0.f;
    for (auto is = spoke_lo; is < spoke_hi; is++) {
      float const scale = R0(ks.chip(is, 2).abs().maximum())();
      auto const calS = GrabSources(ks, scale, is, gap_sz + n_tgt, n_src, n_cal);
      auto const calT = GrabTargets(ks, scale, is, gap_sz, n_tgt, n_cal);
      auto const W = CalcWeights(calS, calT, lambda);
      if (!W.array().allFinite()) {
        log.info(FMT_STRING("Weight calculation error for spoke {}"), is);
        continue;
      }

      rmse_accum += sqrt(((W * calS) - calT).squaredNorm() / calT.size());

      auto const S1 = GrabSourcesOne(ks, scale, is, gap_sz, n_src, 1);
      if (is_simul) {
        FillSimultaneous(S1, W, scale, is, ks);
      } else {
        FillIterative(S1, W, scale, is, gap_sz, ks);
      }
    }
    log.info(
        FMT_STRING("Average rmse spokes {}-{}: {}"),
        spoke_lo,
        spoke_hi,
        rmse_accum / (spoke_hi - spoke_lo));
  };
  Threads::RangeFor(spoke_task, 0, ks.dimension(2));
  Cx0 const k0 = ks.chip(0, 1).mean();
  fmt::print("Average k0: {}\n", k0());
}
