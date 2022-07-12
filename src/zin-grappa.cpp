#include "zin-grappa.hpp"

#include "parse_args.h"
#include "tensorOps.h"
#include "threads.h"

#include <Eigen/SVD>
#include <complex>

namespace rl {
// Helper Functions
Eigen::MatrixXcf GrabSources(
  Cx3 const &ks,
  float const scale,
  Index const n_src,
  Index const s_read,
  Index const n_read,
  std::vector<Index> const &spokes)
{
  assert((s_read + n_read + n_src) < ks.dimension(1));
  Index const n_chan = ks.dimension(0);
  Index const n_spoke = spokes.size();
  Eigen::MatrixXcf S(n_chan * n_src, n_read * n_spoke);
  S.setZero();
  for (Index i_spoke = 0; i_spoke < n_spoke; i_spoke++) {
    Index const col_spoke = i_spoke * n_read;
    Index const ind_spoke = spokes[i_spoke];
    assert(ind_spoke < ks.dimension(2));
    for (Index i_read = 0; i_read < n_read; i_read++) {
      Index const col = col_spoke + i_read;
      for (Index i_coil = 0; i_coil < n_chan; i_coil++) {
        Index const row_coil = i_coil * n_src;
        for (Index i_src = 0; i_src < n_src; i_src++) {
          Index const row = row_coil + i_src;
          S(row, col) = ks(i_coil, s_read + i_read + i_src, ind_spoke) / scale;
        }
      }
    }
  }
  return S;
}

Eigen::MatrixXcf
GrabTargets(Cx3 const &ks, float const scale, Index const s_read, Index const n_read, std::vector<Index> const &spokes)
{
  Index const n_chan = ks.dimension(0);
  Index const n_spoke = spokes.size();
  Eigen::MatrixXcf T(n_chan, n_read * n_spoke);
  T.setZero();
  for (Index i_spoke = 0; i_spoke < n_spoke; i_spoke++) {
    Index const col_spoke = i_spoke * n_read;
    Index const ind_spoke = spokes[i_spoke];
    for (Index i_read = 0; i_read < n_read; i_read++) {
      Index const col = col_spoke + i_read;
      for (Index i_coil = 0; i_coil < n_chan; i_coil++) {
        T(i_coil, col) = ks(i_coil, s_read + i_read, ind_spoke) / scale;
      }
    }
  }
  return T;
}

Eigen::MatrixXcf CalcWeights(Eigen::MatrixXcf const &src, Eigen::MatrixXcf const tgt, float const lambda)
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

std::vector<Index> FindClosest(R3 const &traj, Index const &tgt, Index const &n_spoke, std::vector<Index> &all_spokes)
{
  std::vector<Index> spokes(n_spoke);
  R1 const end_is = traj.chip(tgt, 2).chip(traj.dimension(1) - 1, 1);
  std::partial_sort(
    all_spokes.begin(), all_spokes.begin() + n_spoke, all_spokes.end(), [&traj, end_is](Index const a, Index const b) {
      auto const &end_a = traj.chip(a, 2).chip(traj.dimension(1) - 1, 1);
      auto const &end_b = traj.chip(b, 2).chip(traj.dimension(1) - 1, 1);
      return Norm(end_a - end_is) < Norm(end_b - end_is);
    });
  std::copy_n(all_spokes.begin(), n_spoke, spokes.begin());
  return spokes;
}

// Actual calculation
void zinGRAPPA(
  Index const gap_sz,
  Index const n_src,
  Index const n_spoke,
  Index const n_read1,
  float const lambda,
  R3 const &traj,
  Cx3 &ks)
{
  Index const n_read = n_read1 < 1 ? ks.dimension(1) - (gap_sz + n_src) : n_read1;

  Log::Print(FMT_STRING("ZINFANDEL Gap {} Sources {} Cal Spokes/Read {}/{} "), gap_sz, n_src, n_spoke, n_read);

  for (Index ig = gap_sz; ig > 0; ig--) {
    auto spoke_task = [&](Index const is) {
      std::vector<Index> all_spokes(ks.dimension(2)); // Need a thread-local copy of the indices
      std::iota(all_spokes.begin(), all_spokes.end(), 0L);
      float const scale = R0(ks.chip(is, 2).abs().maximum())();
      auto const spokes = FindClosest(traj, is, n_spoke, all_spokes);
      auto const calS = GrabSources(ks, scale, n_src, ig + 1, n_read, spokes);
      auto const calT = GrabTargets(ks, scale, ig, n_read, spokes);
      auto const W = CalcWeights(calS, calT, lambda);
      auto const S = GrabSources(ks, scale, n_src, ig, 1, {is});
      auto const T = W * S;
      for (Index icoil = 0; icoil < ks.dimension(0); icoil++) {
        ks(icoil, ig - 1, is) = T(icoil) * scale;
      }
    };
    Log::Print(FMT_STRING("Gap index {}"), ig);
    Threads::For(spoke_task, 0, ks.dimension(2), "Spokes");
  }
}
} // namespace rl
