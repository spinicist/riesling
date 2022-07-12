#pragma once

#include "info.h"
#include "log.h"
#include "sense.h"

namespace rl {
/*!
 * ZTE Infilling From Autocallibration NeighbourhooD ELements
 */
void zinGRAPPA(
  Index const gap_sz,
  Index const n_src,
  Index const n_spoke,
  Index const n_read,
  float const lambda,
  R3 const &traj,
  Cx3 &ks);

/*!
 *  Helper functions exposed for testing
 */
Eigen::MatrixXcf GrabSources(
  Cx3 const &ks,
  float const scale,
  Index const n_src,
  Index const s_read,
  Index const n_read,
  std::vector<Index> const &spokes);

Eigen::MatrixXcf GrabTargets(
  Cx3 const &ks,
  float const scale,
  Index const s_read,
  Index const n_read,
  std::vector<Index> const &spokes);
}
