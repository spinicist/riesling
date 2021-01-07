#include "gridder.h"

#include "filter.h"
#include "io_nifti.h"
#include "kaiser-bessel.h"
#include "threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

template <typename T>
inline decltype(auto) nearest(T &&x)
{
  return x.array().unaryExpr([](float const &e) { return static_cast<int16_t>(std::rint(e)); });
}

// Helper function to get a "good" FFT size. Empirical rule of thumb - multiples of 8 work well
long fft_size(float const x)
{
  if (x > 8.f) {
    return (std::lrint(x) + 7L) & ~7L;
  } else {
    return (long)std::ceil(x);
  }
}

Gridder::Gridder(
    RadialInfo const &info,
    R3 const &traj,
    float const os,
    bool const est_dc,
    bool const kb,
    bool const stack,
    Log &log,
    float const res,
    bool const shrink)
    : info_{info}
    , oversample_{os}
    , DCexp_{1.f}
    , log_{log}
{
  assert(traj.dimension(0) == 3);
  assert(traj.dimension(1) == info_.read_points);
  assert(traj.dimension(2) == info_.spokes_total());

  float const ratio = (res > 0.f) ? info_.voxel_size.minCoeff() / res : 1.f;
  long const nomSz = fft_size(oversample_ * info_.matrix.maxCoeff());
  long const gridSz = shrink ? fft_size(nomSz * ratio) : nomSz;
  long const nomRad = nomSz / 2 - 1;
  long const maxRad = gridSz / 2 - 1;
  if (stack) {
    dims_ = {gridSz, gridSz, info_.matrix[2]};
  } else {
    dims_ = {gridSz, gridSz, gridSz};
  }
  log_.info(FMT_STRING("Gridder: Dimensions {}"), dims_);

  interp_ = kb ? (Interpolator *)new KaiserBessel(3, oversample_, dims_, !stack)
               : (Interpolator *)new NearestNeighbour();

  Profile const hires =
      stack ? profile2D(
                  traj.chip(info_.spokes_lo, 2), info_.spokes_hi / dims_[2], nomRad, maxRad, 1.f)
            : profile3D(traj.chip(info_.spokes_lo, 2), info_.spokes_hi, nomRad, maxRad, 1.f);
  coords_ = genCoords(traj, info_.spokes_lo, info_.spokes_hi, hires);
  if (info_.spokes_lo) {
    // Only grid the low-res k-space out to the point the hi-res k-space begins (i.e. fill the
    // dead-time gap)
    float const radialOverSamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
    float const max_r = info_.read_gap * oversample_ / radialOverSamp;
    Profile const lores =
        stack
            ? profile2D(traj.chip(0, 2), info_.spokes_lo / dims_[2], nomRad, max_r, info_.lo_scale)
            : profile3D(traj.chip(0, 2), info_.spokes_lo, nomRad, max_r, info_.lo_scale);
    auto const temp = genCoords(traj, 0, info_.spokes_lo, lores);
    coords_.insert(coords_.end(), temp.begin(), temp.end());
  }
  sortCoords();
  if (est_dc) {
    iterativeDC();
  }
}

inline Point3 toCart(R1 const &p, float const xyScale, float const zScale)
{
  return Point3{p(0) * xyScale, p(1) * xyScale, p(2) * zScale};
}

Gridder::Profile Gridder::profile2D(
    R2 const &traj, long const spokes, long const nomRad, float const maxRad, float const scale)
{
  Profile profile;
  profile.xy = nomRad / scale;
  profile.z = 1.f;
  // Calculate the point spacing
  float const radialOverSamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
  float const k_delta = oversample_ / (radialOverSamp * scale);
  float const V = 2.f * k_delta * M_PI / spokes; // Area element
  // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
  float const R = (M_PI * info_.matrix.maxCoeff()) / (spokes * scale);
  float const flat_start = nomRad / sqrt(R);
  float const flat_val = V * flat_start;

  // Count the number of points per spoke we will retain. Assume spokes have a simple radial
  // profile. Probably excludes stack-of-cones / crazy etch-a-sketch trajectories
  for (int16_t ir = info_.read_gap; ir < info_.read_points; ir++) {
    float const rad = toCart(traj.chip(ir, 1), profile.xy, profile.z).norm();
    if (rad <= maxRad) { // Discard points above the desired resolution
      profile.lo = std::min(ir, profile.lo);
      profile.hi = std::max(ir, profile.hi);
      if (rad == 0.f) {
        profile.DC.push_back(V / 8.f);
      } else if (rad < flat_start) {
        profile.DC.push_back(V * rad);
      } else {
        profile.DC.push_back(flat_val);
      }
    }
  }

  log_.info(
      FMT_STRING("2D profile using points {}-{} (R={}, flat_start {} flat_val {})"),
      profile.lo,
      profile.hi,
      R,
      flat_start,
      flat_val);
  return profile;
}

Gridder::Profile Gridder::profile3D(
    R2 const &traj, long const spokes, long const nomRad, float const maxRad, float const scale)
{
  Profile profile;
  profile.xy = profile.z = nomRad / scale;
  // Calculate the point spacing
  float const radialOverSamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
  float const k_delta = oversample_ / (radialOverSamp * scale);
  float const V = (4.f / 3.f) * k_delta * M_PI / spokes; // Volume element
  // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
  float const R =
      (M_PI * info_.matrix.maxCoeff() * info_.matrix.maxCoeff()) / (spokes * scale * scale);
  float const flat_start = profile.xy / (oversample_ * sqrt(R));
  float const flat_val = V * (3. * (flat_start * flat_start) + 1. / 4.);

  // Count the number of points per spoke we will retain. Assume spokes have a simple radial
  // profile. Probably excludes stack-of-cones / crazy etch-a-sketch trajectories
  for (int16_t ir = info_.read_gap; ir < info_.read_points; ir++) {
    float const rad = toCart(traj.chip(ir, 1), profile.xy, profile.z).norm();
    if (rad <= maxRad) { // Discard points above the desired resolution
      profile.lo = std::min(ir, profile.lo);
      profile.hi = std::max(ir, profile.hi);
      if (rad == 0.f) {
        profile.DC.push_back(V * 1.f / 8.f);
      } else if (rad < flat_start) {
        profile.DC.push_back(V * (3.f * (rad * rad) + 1.f / 4.f));
      } else {
        profile.DC.push_back(flat_val);
      }
    }
  }

  log_.info(
      FMT_STRING("3D profile using points {}-{} (R={}, flat_start {} flat_val {})"),
      profile.lo,
      profile.hi,
      R,
      flat_start,
      flat_val);
  return profile;
}

std::vector<Gridder::Coords>
Gridder::genCoords(R3 const &traj, int32_t const spoke0, long const spokeSz, Profile const &profile)
{
  long const readSz = profile.hi - profile.lo + 1;
  long const kSz = interp_->size();
  std::vector<Coords> coords(readSz * spokeSz * kSz);

  std::fesetround(FE_TONEAREST);
  Size3 wrapSz{dims_[0], dims_[1], dims_[2]}; // Annoying type issue
  auto coordTask = [&](long const lo_spoke, long const hi_spoke) {
    long index = lo_spoke * readSz * kSz;
    for (int32_t is = lo_spoke; is < hi_spoke; is++) {
      for (int16_t ir = profile.lo; ir <= profile.hi; ir++) {
        NoncartesianIndex const nc{.read = ir, .spoke = is + spoke0};
        R1 const tp = traj.chip(nc.spoke, 2).chip(nc.read, 1);
        Point3 const xyz = toCart(tp, profile.xy, profile.z);
        Size3 const cart = nearest(xyz);
        Point3 const offset = xyz - cart.cast<float>().matrix();
        auto const kernel = interp_->weights(offset);
        for (auto &k : kernel) {
          // log_.info(FMT_STRING("k point {} weight {}"), k.point.transpose(), k.weight);
          Size3 const w = wrap(cart + k.point, wrapSz);
          coords[index++] = Coords{.cart = CartesianIndex{w(0), w(1), w(2)},
                                   .noncart = nc,
                                   .DC = profile.DC[ir - profile.lo],
                                   .weight = k.weight};
        }
      }
    }
  };
  auto start = log_.start_time();
  Threads::RangeFor(coordTask, spokeSz);
  log_.stop_time(start, "Calculated grid co-ordinates");
  return coords;
}

void Gridder::sortCoords()
{
  auto const start = log_.start_time();
  sortedIndices_.resize(coords_.size());
  std::iota(sortedIndices_.begin(), sortedIndices_.end(), 0);
  std::sort(sortedIndices_.begin(), sortedIndices_.end(), [=](long const a, long const b) {
    auto const &ac = coords_[a].cart;
    auto const &bc = coords_[b].cart;
    return (ac.z < bc.z) ||
           ((ac.z == bc.z) && ((ac.y < bc.y) || ((ac.y == bc.y) && (ac.x < bc.x))));
  });
  log_.stop_time(start, "Sorting co-ordinates");
}

void Gridder::iterativeDC()
{
  log_.info("Estimating density compensation...");
  Cx3 cart = newGrid1();
  Cx2 W(info_.read_points, info_.spokes_total());
  Cx2 Wp(info_.read_points, info_.spokes_total());

  W.setConstant(1.f);
  for (auto &c : coords_) {
    // W(c.radial[0], c.radial[1]) = std::complex(c.DC, 0.f);
    c.DC = 1.f;
  }

  for (long ii = 0; ii < 8; ii++) {
    cart.setZero();
    Wp.setZero();
    toCartesian(W, cart);
    toRadial(cart, Wp);
    Wp = (Wp.real() > 0.f).select(W / Wp, W); // Avoid divide by zero problems
    float const delta = norm(W - Wp);
    W = Wp;
    if (delta < 1.e-5f) {
      log_.info("DC converged, delta was {}", delta);
      break;
    }
  }

  // Copy to co-ord structure
  for (auto &c : coords_) {
    auto const &nc = c.noncart;
    c.DC = W(nc.read, nc.spoke).real();
  }
  log_.info("Density compensation estimated.");
}

Dims3 Gridder::gridDims() const
{
  return dims_;
}

void Gridder::setDC(float const d)
{
  for (auto &c : coords_) {
    c.DC = d;
  }
}

void Gridder::setDCExponent(float const dce)
{
  DCexp_ = dce;
}

Cx4 Gridder::newGrid() const
{
  return Cx4{info_.channels, dims_[0], dims_[1], dims_[2]};
}

Cx3 Gridder::newGrid1() const
{
  return Cx3{dims_};
}

void Gridder::apodize(Cx3 &image) const
{
  interp_->apodize(image);
}

void Gridder::deapodize(Cx3 &image) const
{
  interp_->deapodize(image);
}

void Gridder::toCartesian(Cx2 const &radial, Cx3 &cart) const
{
  assert(radial.dimension(0) == info_.read_points);
  assert(radial.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == dims_[0]);
  assert(cart.dimension(1) == dims_[1]);
  assert(cart.dimension(2) == dims_[2]);
  assert(sortedIndices_.size() == coords_.size());

  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      auto const &cp = coords_[sortedIndices_[ii]];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &dc = pow(cp.DC, DCexp_);
      std::complex<float> const scale(dc * cp.weight, 0.f);
      cart(c.x, c.y, c.z) += scale * radial(nc.read, nc.spoke);
    }
  };

  auto const &start = log_.start_time();
  Threads::RangeFor(grid_task, sortedIndices_.size());
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
  assert(sortedIndices_.size() == coords_.size());

  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      auto const &cp = coords_[sortedIndices_[ii]];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &dc = pow(cp.DC, DCexp_);
      std::complex<float> const scale(dc * cp.weight, 0.f);
      cart.chip(c.z, 3).chip(c.y, 2).chip(c.x, 1) +=
          radial.chip(nc.spoke, 2).chip(nc.read, 1) * scale;
    }
  };

  auto const &start = log_.start_time();
  Threads::RangeFor(grid_task, sortedIndices_.size());
  log_.stop_time(start, "Radial -> Cartesian");
}

void Gridder::toRadial(Cx3 const &cart, Cx2 &radial) const
{
  assert(radial.dimension(0) == info_.read_points);
  assert(radial.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == dims_[0]);
  assert(cart.dimension(1) == dims_[1]);
  assert(cart.dimension(2) == dims_[2]);

  radial.setZero();
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      auto const &cp = coords_[ii];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      radial(nc.read, nc.spoke) += cart(c.x, c.y, c.z) * cp.weight;
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

  radial.setZero();
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      auto const &cp = coords_[ii];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &weight = radial.chip(nc.read, 2).chip(nc.spoke, 1).constant(cp.weight);
      radial.chip(nc.read, 2).chip(nc.spoke, 1) +=
          cart.chip(c.z, 3).chip(c.y, 2).chip(c.x, 1) * weight;
    }
  };
  auto const &start = log_.start_time();
  Threads::RangeFor(grid_task, coords_.size());
  log_.stop_time(start, "Cartesian -> Radial");
}
