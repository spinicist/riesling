#include "parse_args.h"
#include "threads.h"
#include "zinfandel.h"
#include <Eigen/SVD>
#include <complex>

// Helper Functions
Eigen::MatrixXcf GrabSources(
    Cx3 const &ks,
    float const scale,
    long const n_src,
    long const s_read,
    long const n_read,
    std::vector<long> const &spokes)
{
  assert((s_read + n_read + n_src) < ks.dimension(1));
  long const n_chan = ks.dimension(0);
  long const n_spoke = spokes.size();
  Eigen::MatrixXcf S(n_chan * n_src, n_read * n_spoke);
  S.setZero();
  for (long i_spoke = 0; i_spoke < n_spoke; i_spoke++) {
    long const col_spoke = i_spoke * n_read;
    long const ind_spoke = spokes[i_spoke];
    assert(ind_spoke < ks.dimension(2));
    for (long i_read = 0; i_read < n_read; i_read++) {
      long const col = col_spoke + i_read;
      for (long i_coil = 0; i_coil < n_chan; i_coil++) {
        long const row_coil = i_coil * n_src;
        for (long i_src = 0; i_src < n_src; i_src++) {
          long const row = row_coil + i_src;
          S(row, col) = ks(i_coil, s_read + i_read + i_src, ind_spoke) / scale;
        }
      }
    }
  }
  return S;
}

Eigen::MatrixXcf GrabTargets(
    Cx3 const &ks,
    float const scale,
    long const s_read,
    long const n_read,
    std::vector<long> const &spokes)
{
  long const n_chan = ks.dimension(0);
  long const n_spoke = spokes.size();
  Eigen::MatrixXcf T(n_chan, n_read * n_spoke);
  T.setZero();
  for (long i_spoke = 0; i_spoke < n_spoke; i_spoke++) {
    long const col_spoke = i_spoke * n_read;
    long const ind_spoke = spokes[i_spoke];
    for (long i_read = 0; i_read < n_read; i_read++) {
      long const col = col_spoke + i_read;
      for (long i_coil = 0; i_coil < n_chan; i_coil++) {
        T(i_coil, col) = ks(i_coil, s_read + i_read, ind_spoke) / scale;
      }
    }
  }
  return T;
}

Eigen::MatrixXcf
CalcWeights(Eigen::MatrixXcf const &src, Eigen::MatrixXcf const tgt, float const lambda)
{
  if (lambda > 0.f) {
    auto const reg = lambda * src.norm() * Eigen::MatrixXcf::Identity(src.rows(), src.rows());
    auto const rhs = src * src.adjoint() + reg;
    Eigen::MatrixXcf pinv = rhs.completeOrthogonalDecomposition().pseudoInverse();
    return tgt * src.adjoint() * pinv;
  } else {
    Eigen::MatrixXcf const pinv = src.completeOrthogonalDecomposition().pseudoInverse();
    return tgt * pinv;
  }
}

std::vector<long>
FindClosest(R3 const &traj, long const &tgt, long const &n_spoke, std::vector<long> &all_spokes)
{
  std::vector<long> spokes(n_spoke);
  R1 const end_is = traj.chip(tgt, 2).chip(traj.dimension(1) - 1, 1);
  std::partial_sort(
      all_spokes.begin(),
      all_spokes.begin() + n_spoke,
      all_spokes.end(),
      [&traj, end_is](long const a, long const b) {
        auto const &end_a = traj.chip(a, 2).chip(traj.dimension(1) - 1, 1);
        auto const &end_b = traj.chip(b, 2).chip(traj.dimension(1) - 1, 1);
        return norm(end_a - end_is) < norm(end_b - end_is);
      });
  std::copy_n(all_spokes.begin(), n_spoke, spokes.begin());
  return spokes;
}

// Actual calculation
void zinfandel(
    long const gap_sz,
    long const n_src,
    long const n_spoke,
    long const n_read1,
    float const lambda,
    R3 const &traj,
    Cx3 &ks,
    Log &log)
{
  long const n_read = n_read1 < 1 ? ks.dimension(1) - (gap_sz + n_src) : n_read1;

  log.info(
      FMT_STRING("ZINFANDEL Gap {} Sources {} Cal Spokes/Read {}/{} "),
      gap_sz,
      n_src,
      n_spoke,
      n_read);

  for (long ig = gap_sz; ig > 0; ig--) {
    auto spoke_task = [&](long const spoke_lo, long const spoke_hi) {
      std::vector<long> all_spokes(ks.dimension(2)); // Need a thread-local copy of the indices
      std::iota(all_spokes.begin(), all_spokes.end(), 0L);
      for (auto is = spoke_lo; is < spoke_hi; is++) {
        float const scale = R0(ks.chip(is, 2).abs().maximum())();
        auto const spokes = FindClosest(traj, is, n_spoke, all_spokes);
        auto const calS = GrabSources(ks, scale, n_src, ig + 1, n_read, spokes);
        auto const calT = GrabTargets(ks, scale, ig, n_read, spokes);
        auto const W = CalcWeights(calS, calT, lambda);
        auto const S = GrabSources(ks, scale, n_src, ig, 1, {is});
        auto const T = W * S;
        for (long icoil = 0; icoil < ks.dimension(0); icoil++) {
          ks(icoil, ig - 1, is) = T(icoil) * scale;
        }
      }
    };
    Threads::RangeFor(spoke_task, 0, ks.dimension(2));
  }
}
