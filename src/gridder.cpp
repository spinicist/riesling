#include "gridder.h"

#include "fft3.h"
#include "filter.h"
#include "tensorOps.h"
#include "threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

template <typename T>
inline decltype(auto) nearby(T &&x)
{
  return x.array().unaryExpr([](float const &e) { return std::nearbyint(e); });
}

// Helper function to get a "good" FFT size. Empirical rule of thumb - multiples of 8 work well
inline long fft_size(float const x)
{
  if (x > 8.f) {
    return (std::lrint(x) + 7L) & ~7L;
  } else {
    return (long)std::ceil(x);
  }
}

Gridder::Gridder(
    Info const &info,
    R3 const &traj,
    float const os,
    SDC const sdc,
    Kernel *const kernel,
    bool const stack,
    Log &log,
    float const res,
    bool const shrink)
    : info_{info}
    , oversample_{os}
    , DCexp_{1.f}
    , kernel_{kernel}
    , log_{log}
{
  assert(traj.dimension(0) == 3);
  assert(traj.dimension(1) == info_.read_points);
  assert(traj.dimension(2) == info_.spokes_total());

  // Work out grid dimensions for the given resolution and oversampling
  float const ratio = (res > 0.f) ? info_.voxel_size.minCoeff() / res : 1.f;
  long const nomSz = fft_size(oversample_ * info_.matrix.maxCoeff());
  long const nomRad = nomSz / 2 - 1;
  long const gridSz = shrink ? fft_size(nomSz * ratio) : nomSz;
  if (stack) {
    dims_ = {gridSz, gridSz, info_.matrix[2]};
  } else {
    dims_ = {gridSz, gridSz, gridSz};
  }
  long const maxRad =
      (shrink ? (gridSz / 2 - 1) : std::lrint(nomRad * ratio)) - std::floor(kernel_->radius());
  log_.info(
      FMT_STRING("Gridder: Dimensions {} Resolution {} Ratio {} maxRad {}"),
      dims_,
      res,
      ratio,
      maxRad);

  // Count the number of points per spoke we will retain. Assumes profile is constant across spokes
  Profile const hires =
      stack ? profile2D(
                  traj.chip(info_.spokes_lo, 2), info_.spokes_hi / dims_[2], nomRad, maxRad, 1.f)
            : profile3D(traj.chip(info_.spokes_lo, 2), info_.spokes_hi, nomRad, maxRad, 1.f);
  coords_ = genCoords(traj, info_.spokes_lo, info_.spokes_hi, hires);

  if (info_.spokes_lo) {
    // Only grid lo-res to where hi-res begins (i.e. fill the dead-time gap)
    float const spokeOversamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
    float const max_r = info_.read_gap * oversample_ / spokeOversamp;
    Profile const lores =
        stack
            ? profile2D(traj.chip(0, 2), info_.spokes_lo / dims_[2], nomRad, max_r, info_.lo_scale)
            : profile3D(traj.chip(0, 2), info_.spokes_lo, nomRad, max_r, info_.lo_scale);
    auto const temp = genCoords(traj, 0, info_.spokes_lo, lores);
    coords_.insert(coords_.end(), temp.begin(), temp.end());
  }
  sortCoords();

  if (sdc == SDC::None) {
    // Reset the density compensation
    for (auto &c : coords_) {
      c.DC = 1.f;
    }
  } else if (sdc == SDC::Pipe) {
    // First reset
    for (auto &c : coords_) {
      c.DC = 1.f;
    }
    // Run Pipe's method
    auto const W = SDCPipe(info_, this, kernel_, log_);
    // Copy to co-ords list
    for (auto &c : coords_) {
      auto const &nc = c.noncart;
      c.DC = W(nc.read, nc.spoke).real();
    }
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
  float const spokeOversamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
  float const k_delta = oversample_ / (spokeOversamp * scale);
  float const V = 2.f * k_delta * M_PI / spokes; // Area element
  // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
  float const R = (M_PI * info_.matrix.maxCoeff()) / (spokes * scale);
  float const flat_start = nomRad / sqrt(R);
  float const flat_val = V * flat_start;

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
  float const spokeOversamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
  float const k_delta = oversample_ / (spokeOversamp * scale);
  float const V = (4.f / 3.f) * k_delta * M_PI / spokes; // Volume element
  // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
  float const R =
      (M_PI * info_.matrix.maxCoeff() * info_.matrix.maxCoeff()) / (spokes * scale * scale);
  float const flat_start = profile.xy / (oversample_ * sqrt(R));
  float const flat_val = V * (3. * (flat_start * flat_start) + 1. / 4.);

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
  long const totalSz = readSz * spokeSz;
  std::vector<Coords> coords(totalSz);

  std::fesetround(FE_TONEAREST);
  Size3 const center(dims_[0] / 2, dims_[1] / 2, dims_[2] / 2);
  auto coordTask = [&](long const lo_spoke, long const hi_spoke) {
    long index = lo_spoke * readSz;
    for (int32_t is = lo_spoke; is < hi_spoke; is++) {
      for (int16_t ir = profile.lo; ir <= profile.hi; ir++) {
        NoncartesianIndex const nc{.spoke = is + spoke0, .read = ir};
        R1 const tp = traj.chip(nc.spoke, 2).chip(nc.read, 1);
        Point3 const xyz = toCart(tp, profile.xy, profile.z);
        Size3 const gp = nearby(xyz).cast<int16_t>();
        Point3 const offset = xyz - gp.cast<float>().matrix();
        Size3 const cart = gp + center;
        coords[index++] = Coords{.cart = CartesianIndex{cart(0), cart(1), cart(2)},
                                 .noncart = nc,
                                 .DC = profile.DC[ir - profile.lo],
                                 .offset = offset};
      }
    }
  };
  auto start = log_.now();
  Threads::RangeFor(coordTask, spokeSz);
  log_.debug("Grid co-ord calcs: {}", log_.toNow(start));
  return coords;
}

void Gridder::sortCoords()
{
  auto const start = log_.now();
  sortedIndices_.resize(coords_.size());
  std::iota(sortedIndices_.begin(), sortedIndices_.end(), 0);
  std::sort(sortedIndices_.begin(), sortedIndices_.end(), [&](long const a, long const b) {
    auto const &ac = coords_[a].cart;
    auto const &bc = coords_[b].cart;
    return (ac.z < bc.z) ||
           ((ac.z == bc.z) && ((ac.y < bc.y) || ((ac.y == bc.y) && (ac.x < bc.x))));
  });
  log_.debug("Grid co-ord sorting: {}", log_.toNow(start));
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
  Cx4 g(info_.channels, dims_[0], dims_[1], dims_[2]);
  g.setZero();
  return g;
}

Cx3 Gridder::newGrid1() const
{
  Cx3 g(dims_);
  g.setZero();
  return g;
}

void Gridder::toCartesian(Cx2 const &noncart, Cx3 &cart) const
{
  assert(noncart.dimension(0) == info_.read_points);
  assert(noncart.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == dims_[0]);
  assert(cart.dimension(1) == dims_[1]);
  assert(cart.dimension(2) == dims_[2]);
  assert(sortedIndices_.size() == coords_.size());

  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  Log nullLog;
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      if (lo == 0) {
        log_.progress(ii, hi);
      }
      auto const &cp = coords_[sortedIndices_[ii]];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &dc = std::complex<float>(pow(cp.DC, DCexp_), 0.f);
      cart.slice(Sz3{c.x + st[0], c.y + st[1], c.z + st[2]}, sz) +=
          noncart(nc.read, nc.spoke) * kernel_->kspace(cp.offset).cast<Cx>() * dc;
    }
  };

  auto const &start = log_.now();
  Threads::RangeFor(grid_task, sortedIndices_.size());
  log_.debug("Non-cart -> Cart: {}", log_.toNow(start));
}

void Gridder::toCartesian(Cx3 const &noncart, Cx4 &cart) const
{
  assert(noncart.dimension(0) == info_.channels);
  assert(noncart.dimension(1) == info_.read_points);
  assert(noncart.dimension(2) == info_.spokes_total());
  assert(cart.dimension(0) == info_.channels);
  assert(cart.dimension(1) == dims_[0]);
  assert(cart.dimension(2) == dims_[1]);
  assert(cart.dimension(3) == dims_[2]);
  assert(sortedIndices_.size() == coords_.size());

  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      if (lo == 0) {
        log_.progress(ii, hi);
      }
      auto const &cp = coords_[sortedIndices_[ii]];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &dc = pow(cp.DC, DCexp_);
      Cx4 const nck = noncart.chip(nc.spoke, 2)
                          .chip(nc.read, 1)
                          .reshape(Sz4{info_.channels, 1, 1, 1})
                          .broadcast(Sz4{1, sz[0], sz[1], sz[2]});
      R4 const weights = Tile(kernel_->kspace(cp.offset), info_.channels) * dc;
      cart.slice(
          Sz4{0, c.x + st[0], c.y + st[1], c.z + st[2]},
          Sz4{info_.channels, sz[0], sz[1], sz[2]}) += nck * weights.cast<Cx>();
    }
  };

  auto const &start = log_.now();
  Threads::RangeFor(grid_task, sortedIndices_.size());
  log_.debug("Non-cart -> Cart: {}", log_.toNow(start));
}

void Gridder::toNoncartesian(Cx3 const &cart, Cx2 &noncart) const
{
  assert(noncart.dimension(0) == info_.read_points);
  assert(noncart.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == dims_[0]);
  assert(cart.dimension(1) == dims_[1]);
  assert(cart.dimension(2) == dims_[2]);

  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  Log nullLog;
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      if (lo == 0) {
        log_.progress(ii, hi);
      }
      // for (auto ii = 0; ii < coords_.size(); ii++) {
      auto const &cp = coords_[ii];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &ksl = cart.slice(Sz3{c.x + st[0], c.y + st[1], c.z + st[2]}, sz);
      Cx0 const val = ksl.contract(
          kernel_->kspace(cp.offset).cast<Cx>(),
          Eigen::IndexPairList<
              Eigen::type2indexpair<0, 0>,
              Eigen::type2indexpair<1, 1>,
              Eigen::type2indexpair<2, 2>>());
      noncart.chip(nc.spoke, 1).chip(nc.read, 0) = val;
    }
  };
  auto const &start = log_.now();
  Threads::RangeFor(grid_task, coords_.size());
  log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
}

void Gridder::toNoncartesian(Cx4 const &cart, Cx3 &noncart) const
{
  assert(noncart.dimension(0) == info_.channels);
  assert(noncart.dimension(1) == info_.read_points);
  assert(noncart.dimension(2) == info_.spokes_total());
  assert(cart.dimension(0) == info_.channels);
  assert(cart.dimension(1) == dims_[0]);
  assert(cart.dimension(2) == dims_[1]);
  assert(cart.dimension(3) == dims_[2]);

  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      if (lo == 0) {
        log_.progress(ii, hi);
      }
      auto const &cp = coords_[ii];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &ksl = cart.slice(
          Sz4{0, c.x + st[0], c.y + st[1], c.z + st[2]}, Sz4{info_.channels, sz[0], sz[1], sz[2]});
      R3 const w = kernel_->kspace(cp.offset);
      noncart.chip(nc.spoke, 2).chip(nc.read, 1) = ksl.contract(
          w.cast<Cx>(),
          Eigen::IndexPairList<
              Eigen::type2indexpair<1, 0>,
              Eigen::type2indexpair<2, 1>,
              Eigen::type2indexpair<3, 2>>());
    }
  };
  auto const &start = log_.now();
  Threads::RangeFor(grid_task, coords_.size());
  log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
}
