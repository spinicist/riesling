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

Gridder::Gridder(RadialInfo const &info, R3 const &traj, float const os, bool const stack, Log &log)
    : info_{info}
    , oversample_{os}
    , dc_exp_{1.f}
    , log_{log}
{
  assert(traj.dimension(0) == 3);
  assert(traj.dimension(1) == info_.read_points);
  assert(traj.dimension(2) == info_.spokes_total());
  setup(traj, stack, info_.voxel_size.minCoeff(), false);
}

Gridder::Gridder(
    RadialInfo const &info,
    R3 const &traj,
    float const os,
    bool const stack,
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
  setup(traj, stack, res, shrink);
}

// Helper function to get a "good" FFT size. Empirical rule of thumb - multiples of 8 work well
long fft_size(float const x)
{
  return (std::lrint(x) + 7L) & ~7L;
}

void Gridder::setup(R3 const &traj, bool const stack, float const res, bool const shrink_grid)
{
  float const ratio = info_.voxel_size.minCoeff() / res;
  long const nominalDiameter = fft_size(oversample_ * info_.matrix.maxCoeff());
  long const nominalRadius = nominalDiameter / 2;
  long const gridSz = shrink_grid ? fft_size(nominalDiameter * ratio) : nominalDiameter;
  if (stack) {
    dims_ = {gridSz, gridSz, info_.matrix[2]};
  } else {
    dims_ = {gridSz, gridSz, gridSz};
  }

  log_.info(
      FMT_STRING("Gridder: Desired res {} mm, oversample {}, dims {} nominal size {}"),
      res,
      oversample_,
      dims_,
      nominalRadius);

  std::fesetround(FE_TONEAREST);
  coords_.reserve(info_.read_points * info_.spokes_total());
  auto const mergeLo = MergeLo(info_);
  auto const mergeHi = MergeHi(info_);
  float const xyScale = (float)nominalRadius;
  float const zScale = stack ? 1.f : (float)nominalRadius;
  float const maxRad = nominalRadius * ratio;
  Size3 wrapSz{dims_[0], dims_[1], dims_[2]}; // Annoying type issue
  auto start = log_.start_time();
  for (long is = 0; is < info_.spokes_total(); is++) {
    for (Eigen::Index ir = 0; ir < info_.read_points; ir++) {
      R1 const tp = traj.chip(is, 2).chip(ir, 1);
      Point3 const cart = Point3{tp(0) * xyScale, tp(1) * xyScale, tp(2) * zScale};
      float const rad = stack ? cart.head(2).matrix().norm() : cart.matrix().norm();
      if (rad < maxRad) {
        Size3 const wrapped = wrap(nearest(cart), wrapSz);
        coords_.push_back(CoordSet{.cart = cart,
                                   .wrapped = wrapped,
                                   .radial = {ir, is},
                                   .merge = (is < info_.spokes_lo) ? mergeLo(ir) : mergeHi(ir)});
      }
    }
  };
  log_.stop_time(start, "Calculated grid co-ordinates");
  log_.info("Total co-ordinates {}", coords_.size());
  start = log_.start_time();
  std::sort(coords_.begin(), coords_.end(), [=](CoordSet const &a, CoordSet const &b) {
    auto const &aw = a.wrapped;
    auto const &bw = b.wrapped;
    return (aw[2] < bw[2]) ||
           ((aw[2] == bw[2]) && ((aw[1] < bw[1]) || ((aw[1] == bw[1]) && (aw[0] < bw[0]))));
  });
  log_.stop_time(start, "Sorting co-ordinates");
  analyticDC(stack, nominalRadius);
}

Dims3 Gridder::gridDims() const
{
  return dims_;
}

void Gridder::setDCExponent(float const dce)
{
  dc_exp_ = dce;
}

Cx4 Gridder::newGrid() const
{
  return Cx4{info_.channels, dims_[0], dims_[1], dims_[2]};
}

Cx3 Gridder::newGrid1() const
{
  return Cx3{dims_};
}

void Gridder::toCartesian(Cx2 const &radial, Cx3 &cart) const
{
  assert(radial.dimension(0) == info_.read_points);
  assert(radial.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == dims_[0]);
  assert(cart.dimension(1) == dims_[1]);
  assert(cart.dimension(2) == dims_[2]);

  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      if (cp.merge > 0.f) {
        auto const &icart = cp.wrapped;
        auto const &dc = pow(cp.DC, dc_exp_);
        std::complex<float> const scale(dc * cp.merge, 0.f);
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
  assert(cart.dimension(1) == dims_[0]);
  assert(cart.dimension(2) == dims_[1]);
  assert(cart.dimension(3) == dims_[2]);

  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      if (cp.merge > 0.f) {
        auto const &icart = cp.wrapped;
        auto const &dc = pow(cp.DC, dc_exp_);
        std::complex<float> const scale(dc * cp.merge, 0.f);
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
  assert(radial.dimension(0) == info_.read_points);
  assert(radial.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == dims_[0]);
  assert(cart.dimension(1) == dims_[1]);
  assert(cart.dimension(2) == dims_[2]);

  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      if (cp.merge > 0.f) {
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
  assert(radial.dimension(1) == info_.read_points);
  assert(radial.dimension(2) == info_.spokes_total());
  assert(cart.dimension(1) == dims_[0]);
  assert(cart.dimension(2) == dims_[1]);
  assert(cart.dimension(3) == dims_[2]);

  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      if (cp.merge > 0.f) {
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

void Gridder::analyticDC(bool const stack, long const nominalRad)
{
  if (stack) {
    log_.info("Stack-type analytic DC...");
    // Work out area element
    auto const delta = 1.;
    float const d_lo = 2 * M_PI * delta * delta / info_.spokes_lo;
    float const d_hi = 2 * M_PI * delta * delta / info_.spokes_hi;

    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const approx_undersamp = M_PI * info_.matrix.maxCoeff() / info_.spokes_hi;
    float const flat_start = nominalRad / sqrt(approx_undersamp);
    float const flat_val = d_hi * flat_start;

    for (auto &c : coords_) {
      auto const &irad = c.radial;
      float const k_r = c.cart.head(2).matrix().norm(); // xy only
      auto const &d_k = irad(1) < info_.spokes_lo ? d_lo : d_hi;
      if (k_r == 0.f) {
        c.DC = d_k * 1.f / 8.f;
      } else if (k_r < flat_start) {
        c.DC = d_k * k_r;
      } else {
        c.DC = flat_val;
      }
    }
  } else {
    log_.info("Analytic DC...");
    // Work out volume element
    auto const delta = 1.;
    float const d_lo = (4. / 3.) * M_PI * delta * delta * delta / info_.spokes_lo;
    float const d_hi = (4. / 3.) * M_PI * delta * delta * delta / info_.spokes_hi;

    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const approx_undersamp =
        (M_PI * info_.matrix.maxCoeff() * info_.matrix.maxCoeff()) / info_.spokes_hi;
    float const flat_start = nominalRad / sqrt(approx_undersamp);
    float const flat_val = d_hi * (3. * (flat_start * flat_start) + 1. / 4.);

    for (auto &c : coords_) {
      auto const &irad = c.radial;
      float const k_r = c.cart.matrix().norm(); // This has already been multipled by nominalRadius_
      auto const &d_k = irad(1) < info_.spokes_lo ? d_lo : d_hi;
      if (k_r == 0.f) {
        c.DC = d_k * 1.f / 8.f;
      } else if (k_r < flat_start) {
        c.DC = d_k * (3. * (k_r * k_r) + 1. / 4.);
      } else {
        c.DC = flat_val;
      }
    }
  }
}

void Gridder::estimateDC()
{
  log_.info("Estimating density compensation...");
  Cx3 cart(gridDims());
  Cx2 W(info_.read_points, info_.spokes_total());
  Cx2 Wp(info_.read_points, info_.spokes_total());

  // Reset DC
  for (auto &c : coords_) {
    c.DC = 1.f;
  }

  W.setConstant(1.0f);
  for (long ii = 0; ii < 16; ii++) {
    cart.setZero();
    Wp.setConstant(1.0f); // Avoid division by zero
    toCartesian(W, cart);
    toRadial(cart, Wp);
    W = W / Wp;
  }

  // Copy to co-ord structure
  for (auto &c : coords_) {
    c.DC = W(c.radial[0], c.radial[1]).real();
  }

  log_.info("Density compensation estimated.");
}
