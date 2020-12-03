#include "parse_args.h"
#include "threads.h"
#include "zinfandel.h"
#include <Eigen/SVD>
#include <complex>

using namespace std::complex_literals;

Eigen::MatrixXcd GrabSources(
    Cx3 const &ks,
    float const scale,
    long const n_src,
    long const s_spoke,
    long const n_spoke1,
    long const s_read,
    long const n_read)
{
  long const n_chan = ks.dimension(0);
  long const lo_spoke = std::max(0L, s_spoke - (long)std::floor(n_spoke1 / 2.f));
  long const hi_spoke = std::min(ks.dimension(2), s_spoke + (long)std::ceil(n_spoke1 / 2.f));
  long const n_spoke = hi_spoke - lo_spoke;
  Eigen::MatrixXcd S(n_chan * n_src, n_read * n_spoke);
  S.setZero();
  for (long i_spoke = lo_spoke; i_spoke < hi_spoke; i_spoke++) {
    long const col_spoke = (i_spoke - lo_spoke) * n_read;
    for (long i_read = 0; i_read < n_read; i_read++) {
      long const col = col_spoke + i_read;
      for (long i_coil = 0; i_coil < n_chan; i_coil++) {
        long const row_coil = i_coil * n_src;
        for (long i_src = 0; i_src < n_src; i_src++) {
          long const row = row_coil + i_src;
          S(row, col) = ks(i_coil, s_read + i_read + i_src, i_spoke) / scale;
        }
      }
    }
  }
  return S;
}

Eigen::MatrixXcd GrabTargets(
    Cx3 const &ks,
    float const scale,
    long const n_tgt,
    long const s_spoke,
    long const n_spoke1,
    long const s_read,
    long const n_read)
{
  long const n_chan = ks.dimension(0);
  long const lo_spoke = std::max(0L, s_spoke - (long)std::floor(n_spoke1 / 2.f));
  long const hi_spoke = std::min(ks.dimension(2), s_spoke + (long)std::ceil(n_spoke1 / 2.f));
  long const n_spoke = hi_spoke - lo_spoke;
  Eigen::MatrixXcd T(n_chan * n_tgt, n_read * n_spoke);
  T.setZero();
  for (long i_spoke = lo_spoke; i_spoke < hi_spoke; i_spoke++) {
    long const col_spoke = (i_spoke - lo_spoke) * n_read;
    for (long i_read = 0; i_read < n_read; i_read++) {
      long const col = col_spoke + i_read;
      for (long i_coil = 0; i_coil < n_chan; i_coil++) {
        long const row_coil = i_coil * n_tgt;
        for (long i_tgt = 0; i_tgt < n_tgt; i_tgt++) {
          long const row = row_coil + i_tgt;
          T(row, col) = ks(i_coil, s_read + i_read + i_tgt, i_spoke) / scale;
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
  long const n_chan = ks.dimension(0);
  long const n_rows = S.rows();
  long const n_src = n_rows / n_chan;
  Eigen::VectorXcd Si = S;
  Eigen::VectorXcd T(n_chan);
  for (long ii = 0; ii < n_tgt; ii++) {
    long const ir = n_tgt - 1 - ii;
    double const w0 = (ir == 0) ? 0.5 : 1.;
    T = W * Si;
    for (long icoil = 0; icoil < n_chan; icoil++) {
      ks(icoil, ir, spoke) = T(icoil) * (w0 * scale);
    }
    Si.tail(n_rows - 1) = Si.head(n_rows - 1).eval(); // Shift everything along, ensure temporary
    Eigen::VectorXcd check = Si(Eigen::seqN(0, n_chan, n_src));
    Si(Eigen::seqN(0, n_chan, n_src)) = T; // Copy in new values from T
  }
}

void FillSimultaneous(
    Eigen::VectorXcd const &S,
    Eigen::MatrixXcd const &W,
    float const scale,
    long const spoke,
    Cx3 &ks)
{
  long const n_chan = ks.dimension(0);
  long const n_tgt = W.rows() / n_chan;
  auto const T = W * S;
  for (long itgt = 0; itgt < n_tgt; itgt++) {
    double const w0 = (itgt == 0) ? 0.5 : 1.;
    for (long icoil = 0; icoil < n_chan; icoil++) {
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
    long const gap_sz,
    long const n_tgt,
    long const n_src,
    long const n_spoke,
    long const n_read1,
    float const lambda,
    Cx3 &ks,
    Log &log)
{
  long const n_read = n_read1 < 1 ? ks.dimension(1) - (gap_sz + n_tgt + n_src - 1) : n_read1;

  log.info(
      FMT_STRING("ZINFANDEL Targets {} Sources {} Cal Spokes/Read {}/{} Gap {}"),
      n_tgt,
      n_src,
      n_spoke,
      n_read,
      gap_sz);

  auto spoke_task = [&](long const spoke_lo, long const spoke_hi) {
    float rmse_accum = 0.f;
    for (auto is = spoke_lo; is < spoke_hi; is++) {
      float const scale = R0(ks.chip(is, 2).abs().maximum())();
      auto const calS = GrabSources(ks, scale, n_src, is, n_spoke, gap_sz + n_tgt, n_read);
      auto const calT = GrabTargets(ks, scale, n_tgt, is, n_spoke, gap_sz, n_read);
      auto const W = CalcWeights(calS, calT, lambda);
      if (!W.array().allFinite()) {
        log.info(FMT_STRING("Weight calculation error for spoke {}"), is);
        continue;
      }

      rmse_accum += ((W * calS) - calT).squaredNorm() / calT.size();
      auto const S = GrabSources(ks, scale, n_src, is, 1, gap_sz, 1);
      if (n_tgt > 1) {
        FillSimultaneous(S, W, scale, is, ks);
      } else {
        FillIterative(S, W, scale, is, gap_sz, ks);
      }
    }
    float const rmse_error = sqrt(rmse_accum / (spoke_hi - spoke_lo));
    log.info(FMT_STRING("Average rmse spokes {}-{}: {}"), spoke_lo, spoke_hi, rmse_error);
  };
  // spoke_task(0, ks.dimension(2));
  Threads::RangeFor(spoke_task, 0, ks.dimension(2));
}
