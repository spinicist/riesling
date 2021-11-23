#include "grid-kb.h"

#include "../cropper.h"
#include "../fft_plan.h"
#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

template <int InPlane, int ThroughPlane>
GridKB<InPlane, ThroughPlane>::GridKB(
  Trajectory const &traj,
  float const os,
  bool const unsafe,
  Log &log,
  float const inRes,
  bool const shrink)

  template <int InPlane, int ThroughPlane>
  GridKB<InPlane, ThroughPlane>::GridKB(Mapping const &mapping, bool const unsafe, Log &log)

    template <int InPlane, int ThroughPlane>
    void GridKB<InPlane, ThroughPlane>::Adj(Cx3 const &noncart, Cx4 &cart) const

  template <int InPlane, int ThroughPlane>
  void GridKB<InPlane, ThroughPlane>::A(Cx4 const &cart, Cx3 &noncart) const

  template <int InPlane, int ThroughPlane>
  R3 GridKB<InPlane, ThroughPlane>::apodization(Sz3 const sz) const

  template struct GridKB<3, 1>;
template struct GridKB<3, 3>;
template struct GridKB<5, 1>;
template struct GridKB<5, 5>;
