#include "gridder.h"

#include "filter.h"
#include "io_nifti.h"
#include "threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

template <typename T>
inline decltype(auto) nearest(T &&x)
{
  return x.unaryExpr([](float const &e) { return std::lrintf(e); });
}

Gridder::Gridder(RadialInfo const &info, R3 const &traj, float const os, Log &log)
    : info_{info}
    , oversample_{os}
    , dc_exp_{1.f}
    , log_{log}
{
  assert(traj.dimension(0) == 3);
  assert(traj.dimension(1) == info_.read_points);
  assert(traj.dimension(2) == info_.spokes_total());
  setup(traj, info_.voxel_size.minCoeff(), false);
}

Gridder::Gridder(
    RadialInfo const &info,
    R3 const &traj,
    float const os,
    float const res,
    bool const shrink,
    Log &log)
    : info_{info}
    , oversample_{os}
    , dc_exp_{1.f}
    , log_{log}
{
  assert(traj.dimension(0) == 3);
  assert(traj.dimension(1) == info_.read_points);
  assert(traj.dimension(2) == info_.spokes_total());
  setup(traj, res, shrink);
}

void Gridder::setup(R3 const &traj, float const res, bool const shrink_grid)
{
  float const ratio = info_.voxel_size.minCoeff() / res;
  highestReadIndex_ = ratio * info_.read_points;
  if (shrink_grid) {
    fullSize_ = std::ceil(ratio * oversample_ * info_.matrix.maxCoeff());
    nominalSize_ = std::ceil(oversample_ * info_.matrix.maxCoeff()) / 2;
  } else {
    fullSize_ = std::ceil(oversample_ * info_.matrix.maxCoeff());
    nominalSize_ = fullSize_ / 2;
  }
  log_.info(
      FMT_STRING(
          "Gridder: Desired res {} mm, oversample {}, grid size {} nominal size {}, high index {}"),
      res,
      oversample_,
      fullSize_,
      nominalSize_,
      highestReadIndex_);

  coords_.resize(highestReadIndex_ * info_.spokes_total());
  merge_ = R2(highestReadIndex_, info_.spokes_total());
  merge_.setZero();
  auto mergeLo = MergeLo(info_);
  auto mergeHi = MergeHi(info_);

  std::fesetround(FE_TONEAREST);
  auto spoke_task = [&](long const spoke) {
    long index = spoke * highestReadIndex_;
    for (Eigen::Index ir = 0; ir < highestReadIndex_; ir++) {
      R1 const tp = traj.chip(spoke, 2).chip(ir, 1);
      Point3 const cart = Point3{tp(0), tp(1), tp(2)} * nominalSize_;
      Size3 const wrapped = wrap(nearest(cart), fullSize_);
      coords_[index++] = CoordSet{.cart = cart, .wrapped = wrapped, .radial = {ir, spoke}};
      if (spoke < info_.spokes_lo) {
        merge_(ir, spoke) = mergeLo(ir);
      } else {
        merge_(ir, spoke) = mergeHi(ir);
      }
    }
  };

  auto start = log_.start_time();
  Threads::For(spoke_task, info_.spokes_total());
  log_.stop_time(start, "Calculating grid co-ordinates");
  start = log_.start_time();

  std::sort(coords_.begin(), coords_.end(), [=](CoordSet const &a, CoordSet const &b) {
    auto const &aw = a.wrapped;
    auto const &bw = b.wrapped;
    return (aw[2] < bw[2]) ||
           ((aw[2] == bw[2]) && ((aw[1] < bw[1]) || ((aw[1] == bw[1]) && (aw[0] < bw[0]))));
  });
  log_.stop_time(start, "Sorting co-ordinates");
  analyticDC();
}

long Gridder::gridRadius() const
{
  return (static_cast<float>(highestReadIndex_) / info_.read_points) * nominalSize_;
}

long Gridder::gridSize() const
{
  return fullSize_;
}

Dims3 const Gridder::gridDims() const
{
  return Dims3{fullSize_, fullSize_, fullSize_};
}

void Gridder::setDCExponent(float const dce)
{
  dc_exp_ = dce;
}

Cx4 Gridder::newGrid() const
{
  return Cx4{info_.channels, fullSize_, fullSize_, fullSize_};
}

Cx3 Gridder::newGrid1() const
{
  return Cx3{fullSize_, fullSize_, fullSize_};
}

void Gridder::toCartesian(Cx2 const &radial, Cx3 &cart) const
{
  assert(radial.dimension(0) == info_.read_points);
  assert(radial.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == fullSize_);
  assert(cart.dimension(1) == fullSize_);
  assert(cart.dimension(2) == fullSize_);

  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      auto const &merge = merge_(iradial(0), iradial(1));
      if (merge > 0.f) {
        auto const &icart = cp.wrapped;
        auto const &dc = pow(DC_(iradial(0), iradial(1)), dc_exp_);
        std::complex<float> const scale(dc * merge, 0.f);
        cart(icart(0), icart(1), icart(2)) += scale * radial(iradial(0), iradial(1));
      }
    }
  };

  auto const &start = log_.start_time();
  Threads::RangeFor(grid_task, coords_.size());
  log_.stop_time(start, "Radial -> Cartesian");
}

void Gridder::toCartesian(Cx3 const &radial, Cx4 &cart) const
{
  assert(radial.dimension(0) == info_.channels);
  assert(radial.dimension(1) == info_.read_points);
  assert(radial.dimension(2) == info_.spokes_total());
  assert(cart.dimension(0) == info_.channels);
  assert(cart.dimension(1) == fullSize_);
  assert(cart.dimension(2) == fullSize_);
  assert(cart.dimension(3) == fullSize_);

  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      auto const &merge = merge_(iradial(0), iradial(1));
      if (merge > 0.f) {
        auto const &icart = cp.wrapped;
        auto const &dc = pow(DC_(iradial(0), iradial(1)), dc_exp_);
        std::complex<float> const scale(dc * merge, 0.f);
        cart.chip(icart[2], 3).chip(icart[1], 2).chip(icart[0], 1) +=
            radial.chip(iradial[1], 2).chip(iradial[0], 1) * scale;
      }
    }
  };

  auto const &start = log_.start_time();
  Threads::RangeFor(grid_task, coords_.size());
  log_.stop_time(start, "Radial -> Cartesian");
}

void Gridder::toRadial(Cx3 const &cart, Cx2 &radial) const
{
  assert(radial.dimension(0) >= highestReadIndex_);
  assert(radial.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == fullSize_);
  assert(cart.dimension(1) == fullSize_);
  assert(cart.dimension(2) == fullSize_);

  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      auto const &merge = merge_(iradial(0), iradial(1));
      if (merge > 0.f) {
        auto const &iw = cp.wrapped;
        radial(iradial(0), iradial(1)) = cart(iw[0], iw[1], iw[2]);
      }
    }
  };
  auto const &start = log_.start_time();
  Threads::RangeFor(grid_task, coords_.size());
  log_.stop_time(start, "Cartesian -> Radial");
}

void Gridder::toRadial(Cx4 const &cart, Cx3 &radial) const
{
  assert(radial.dimension(0) == cart.dimension(0));
  assert(radial.dimension(1) >= highestReadIndex_);
  assert(radial.dimension(2) == info_.spokes_total());
  assert(cart.dimension(1) == fullSize_);
  assert(cart.dimension(2) == fullSize_);
  assert(cart.dimension(3) == fullSize_);

  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      auto const &merge = merge_(iradial(0), iradial(1));
      if (merge > 0.f) {
        auto const &iw = cp.wrapped;
        radial.chip(iradial(1), 2).chip(iradial(0), 1) =
            cart.chip(iw[2], 3).chip(iw[1], 2).chip(iw[0], 1);
      }
    }
  };
  auto const &start = log_.start_time();
  Threads::RangeFor(grid_task, coords_.size());
  log_.stop_time(start, "Cartesian -> Radial");
}

void Gridder::analyticDC()
{
  // Work out volume element
  auto const delta = 1.;
  float const d_lo = (4. / 3.) * M_PI * delta * delta * delta / info_.spokes_lo;
  float const d_hi = (4. / 3.) * M_PI * delta * delta * delta / info_.spokes_hi;

  // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
  float const approx_undersamp =
      (M_PI * info_.matrix.maxCoeff() * info_.matrix.maxCoeff()) / info_.spokes_hi;
  float const flat_start = nominalSize_ / sqrt(approx_undersamp);
  float const flat_val = d_hi * (3. * (flat_start * flat_start) + 1. / 4.);

  DC_ = R2(highestReadIndex_, info_.spokes_total());
  for (auto const &c : coords_) {
    auto const &irad = c.radial;
    float const k_r = c.cart.matrix().norm(); // This has already been multipled by nominalSize_
    auto &entry = DC_(irad(0), irad(1));
    auto const &d_k = irad(1) < info_.spokes_lo ? d_lo : d_hi;
    if (k_r == 0.f) {
      entry = d_k * 1.f / 8.f;
    } else if (k_r < flat_start) {
      entry = d_k * (3. * (k_r * k_r) + 1. / 4.);
    } else {
      entry = flat_val;
    }
  }
}

void Gridder::estimateDC()
{
  log_.info("Estimating density compensation...");
  Cx3 cart(gridDims());
  Cx2 W(info_.read_points, info_.spokes_total());
  Cx2 Wp(info_.read_points, info_.spokes_total());
  DC_.setConstant(1.0f);
  W.setConstant(1.0f);
  for (long ii = 0; ii < 16; ii++) {
    cart.setZero();
    Wp.setConstant(1.0f); // Avoid division by zero
    toCartesian(W, cart);
    toRadial(cart, Wp);
    W = W / Wp;
  }
  DC_ = W.slice(Sz2{0, 0}, Sz2{highestReadIndex_, info_.spokes_total()}).abs();
  log_.info("Density compensation estimated.");
}
