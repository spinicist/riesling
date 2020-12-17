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
  return x.array().unaryExpr([](float const &e) { return std::lrintf(e); });
}

// Helper function to get a "good" FFT size. Empirical rule of thumb - multiples of 8 work well
long fft_size(float const x)
{
  if (x > 8.f) {
    return (std::lrint(x) + 7L) & ~7L;
  } else {
    return (long)std::ceilf(x);
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

  if (kb) {
    kbW_ = 3;
    kbBeta_ = M_PI * sqrtf(pow(kbW_ * (oversample_ - 0.5f) / oversample_, 2.f) - 0.8f);
    kbApodization(stack);
    log_.info(FMT_STRING("Kaiser-Bessel width {} beta {}"), kbW_, kbBeta_);
  } else {
    kbW_ = 0;
    kbBeta_ = 0.f;
  }

  Eigen::TensorMap<R3 const> trajHi(
      &traj(0, 0, info_.spokes_lo), 3, info_.read_points, info_.spokes_hi);
  coords_ = stack ? stackCoords(trajHi, info_.spokes_lo, nomRad, maxRad, est_dc)
                  : fullCoords(trajHi, info_.spokes_lo, nomRad, maxRad, 1.f);
  if (info_.spokes_lo) {
    Eigen::TensorMap<R3 const> trajLo(&traj(0, 0, 0), 3, info_.read_points, info_.spokes_lo);
    auto const temp = stack ? stackCoords(trajLo, 0, nomRad, maxRad, est_dc)
                            : fullCoords(trajLo, 0, nomRad, maxRad, info_.lo_scale);
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

std::vector<Gridder::Coords> Gridder::stackCoords(
    Eigen::TensorMap<R3 const> const &traj,
    long const spokeOffset,
    long const nomRad,
    float const maxRad,
    float const scale)
{
  log_.info("Stack-of-stars type trajectory");
  float const xyRadius = nomRad / scale;
  float const zScale = 1.f;
  // Count the number of points per spoke we will retain. Assume spokes have a simple radial
  // profile. Probably excludes stack-of-cones / crazy etch-a-sketch trajectories
  long read_lo = info_.read_points, read_hi = 0, readSz = 0;
  long hi_r = (scale == 1.f) ? info_.read_points : info_.read_gap * scale;
  for (long ir = info_.read_gap; ir < hi_r; ir++) {
    float const rad = toCart(traj.chip(0, 2).chip(ir, 1), xyRadius, zScale).norm();
    if (rad <= maxRad) { // Discard points above the desired resolution
      read_lo = std::min(ir, read_lo);
      read_hi = std::max(ir, read_hi);
      readSz++;
    }
  }
  long const spokeSz = traj.dimension(2);
  long const kSz = (kbW_ == 0) ? 1 : kbW_ * kbW_;
  std::vector<Coords> coords(readSz * spokeSz * kSz);

  // Calculate the point spacing
  float const radialOverSamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
  float const k_delta = oversample_ / (radialOverSamp * scale);
  float const V = 2.f * k_delta * M_PI / spokeSz; // Volume element
  // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
  float const R = (M_PI * info_.matrix.maxCoeff()) / (spokeSz * scale);
  float const flat_start = nomRad / sqrt(R);
  float const flat_val = V * flat_start;

  auto analyticDC = [&](float const r) {
    if (r == 0.f) {
      return V * 1.f / 8.f;
    } else if (r < flat_start) {
      return V * r;
    } else {
      return flat_val;
    }
  };

  log_.info(
      FMT_STRING("Using points {}-{} on {} spokes, total {} (R={}, flat_start {} flat_val {})"),
      read_lo,
      read_hi,
      spokeSz,
      coords.size(),
      R,
      flat_start,
      flat_val);

  std::fesetround(FE_TONEAREST);
  Size3 wrapSz{dims_[0], dims_[1], dims_[2]}; // Annoying type issue
  auto coordTask = [&](long const lo_spoke, long const hi_spoke) {
    long index = lo_spoke * readSz;
    for (long is = lo_spoke; is < hi_spoke; is++) {
      for (Eigen::Index ir = read_lo; ir <= read_hi; ir++) {
        Point3 const xyz = toCart(traj.chip(is, 2).chip(ir, 1), xyRadius, zScale);
        Size3 const cart = nearest(xyz);
        float const DC = analyticDC(xyz.head(2).norm());
        Size2 const rad = {ir, is + spokeOffset};
        if (kbW_ == 0) { // Nearest-neighbour
          Size3 const wrapped = wrap(cart, wrapSz);
          coords[index++] = Coords{.cart = wrapped, .radial = rad, .DC = DC, .weight = 1.f};
        } else { // Kaiser-Bessel
          Point2 const offset = xyz.head(2) - cart.cast<float>().matrix().head(2);
          R2 const K = KBKernel(offset, kbW_, kbBeta_);
          for (long iy = 0; iy < kbW_; iy++) {
            for (long ix = 0; ix < kbW_; ix++) {
              Size3 const k = cart + Size3{ix - kbW_ / 2, iy - kbW_ / 2, 0};
              Size3 const w = wrap(k, wrapSz);
              coords[index++] = Coords{.cart = w, .radial = rad, .DC = DC, .weight = K(ix, iy)};
            }
          }
        }
      }
    }
  };
  auto start = log_.start_time();
  Threads::RangeFor(coordTask, traj.dimension(2));
  log_.stop_time(start, "Calculated grid co-ordinates");
  return coords;
}

std::vector<Gridder::Coords> Gridder::fullCoords(
    Eigen::TensorMap<R3 const> const &traj,
    long const spokeOffset,
    long const nomRad,
    float const maxRad,
    float const scale)
{
  log_.info("Full radial type trajectory");
  float const radius = nomRad / scale;
  // Count the number of points per spoke we will retain. Assume spokes have a simple radial
  // profile. Probably excludes stack-of-cones / crazy etch-a-sketch trajectories
  long read_lo = info_.read_points, read_hi = 0, readSz = 0;
  long hi_r = (scale == 1.f) ? info_.read_points : info_.read_gap * scale;
  for (long ir = info_.read_gap; ir < hi_r; ir++) {
    float const rad = toCart(traj.chip(0, 2).chip(ir, 1), radius, radius).norm();
    if (rad <= maxRad) { // Discard points above the desired resolution
      read_lo = std::min(ir, read_lo);
      read_hi = std::max(ir, read_hi);
      readSz++;
    }
  }
  long const spokeSz = traj.dimension(2);
  long const kSz = (kbW_ == 0) ? 1 : kbW_ * kbW_ * kbW_;
  std::vector<Coords> coords(readSz * spokeSz * kSz);

  // Calculate the point spacing
  float const radialOverSamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
  float const k_delta = oversample_ / (radialOverSamp * scale);
  float const V = (4.f / 3.f) * k_delta * M_PI / spokeSz; // Volume element
  // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
  float const R =
      (M_PI * info_.matrix.maxCoeff() * info_.matrix.maxCoeff()) / (spokeSz * scale * scale);
  float const flat_start = radius / (oversample_ * sqrt(R));
  float const flat_val = V * (3. * (flat_start * flat_start) + 1. / 4.);

  auto analyticDC = [&](float const r) -> float {
    if (r == 0.f) {
      return V * 1.f / 8.f;
    } else if (r < flat_start) {
      return V * (3.f * (r * r) + 1.f / 4.f);
    } else {
      return flat_val;
    }
  };

  log_.info(
      FMT_STRING("Using points {}-{} on {} spokes, total {} (R={}, flat_start {} flat_val {})"),
      read_lo,
      read_hi,
      spokeSz,
      coords.size(),
      R,
      flat_start,
      flat_val);

  std::fesetround(FE_TONEAREST);
  Size3 wrapSz{dims_[0], dims_[1], dims_[2]}; // Annoying type issue
  auto coordTask = [&](long const lo_spoke, long const hi_spoke) {
    long index = lo_spoke * readSz * kSz;
    for (long is = lo_spoke; is < hi_spoke; is++) {
      for (Eigen::Index ir = read_lo; ir <= read_hi; ir++) {
        R1 const tp = traj.chip(is, 2).chip(ir, 1);
        Point3 const xyz = toCart(traj.chip(is, 2).chip(ir, 1), radius, radius);
        Size3 const cart = nearest(xyz);
        float const DC = analyticDC(xyz.norm());
        Size2 const rad = {ir, is + spokeOffset};
        if (kbW_ == 0) { // Nearest-neighbour
          Size3 const wrapped = wrap(cart, wrapSz);
          coords[index++] = Coords{.cart = wrapped, .radial = rad, .DC = DC, .weight = 1.f};
        } else { // Kaiser-Bessel
          Point3 const offset = xyz - cart.cast<float>().matrix();
          R3 const K = KBKernel(offset, kbW_, kbBeta_);
          for (long iz = 0; iz < kbW_; iz++) {
            for (long iy = 0; iy < kbW_; iy++) {
              for (long ix = 0; ix < kbW_; ix++) {
                Size3 const k = cart + Size3{ix - kbW_ / 2, iy - kbW_ / 2, iz - kbW_ / 2};
                Size3 const w = wrap(k, wrapSz);
                coords[index++] =
                    Coords{.cart = w, .radial = rad, .DC = DC, .weight = K(ix, iy, iz)};
              }
            }
          }
        }
      }
    }
  };
  auto start = log_.start_time();
  Threads::RangeFor(coordTask, traj.dimension(2));
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
    return (ac[2] < bc[2]) ||
           ((ac[2] == bc[2]) && ((ac[1] < bc[1]) || ((ac[1] == bc[1]) && (ac[0] < bc[0]))));
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
    auto const &ind = c.radial;
    c.DC = W(ind[0], ind[1]).real();
  }
  log_.info("Density compensation estimated.");
}

void Gridder::kbApodization(bool const stack)
{
  float const scale = KB_FT(0., kbBeta_);
  auto apod = [&](int const sz) {
    Eigen::ArrayXf r(sz);
    for (long ii = 0; ii < sz; ii++) {
      float const pos = (ii - sz / 2.f) / static_cast<float>(sz);
      r(ii) = KB_FT(pos * kbW_, kbBeta_) / scale;
    }
    return r;
  };
  apodX_ = apod(dims_[0]);
  apodY_ = apod(dims_[1]);
  if (stack) {
    apodZ_ = Eigen::ArrayXf::Ones(dims_[2]);
  } else {
    apodZ_ = apod(dims_[2]);
  }
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
  if (kbW_ > 0) {
    long const stz = (apodZ_.rows() - image.dimension(2)) / 2;
    long const sty = (apodY_.rows() - image.dimension(1)) / 2;
    long const stx = (apodX_.rows() - image.dimension(0)) / 2;
    for (Eigen::Index iz = 0; iz < image.dimension(2); iz++) {
      float const rz = apodZ_(stz + iz);
      for (Eigen::Index iy = 0; iy < image.dimension(1); iy++) {
        float const ry = apodY_(sty + iy) * rz;
        for (Eigen::Index ix = 0; ix < image.dimension(0); ix++) {
          float const rx = apodX_(stx + ix) * ry;
          image(ix, iy, iz) /= rx;
        }
      }
    }
  }
}

void Gridder::deapodize(Cx3 &image) const
{
  if (kbW_ > 0) {
    long const stz = (apodZ_.rows() - image.dimension(2)) / 2;
    long const sty = (apodY_.rows() - image.dimension(1)) / 2;
    long const stx = (apodX_.rows() - image.dimension(0)) / 2;
    for (Eigen::Index iz = 0; iz < image.dimension(2); iz++) {
      float const rz = apodZ_(stz + iz);
      for (Eigen::Index iy = 0; iy < image.dimension(1); iy++) {
        float const ry = apodY_(sty + iy) * rz;
        for (Eigen::Index ix = 0; ix < image.dimension(0); ix++) {
          float const rx = apodX_(stx + ix) * ry;
          image(ix, iy, iz) *= rx;
        }
      }
    }
  }
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
      auto const &cp = coords_[sortedIndices_[ic]];
      auto const &iradial = cp.radial;

      auto const &icart = cp.cart;
      auto const &dc = pow(cp.DC, DCexp_);
      std::complex<float> const scale(dc * cp.weight, 0.f);
      cart(icart(0), icart(1), icart(2)) += scale * radial(iradial(0), iradial(1));
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
      auto const &cp = coords_[sortedIndices_[ic]];
      auto const &iradial = cp.radial;
      auto const &icart = cp.cart;
      auto const &dc = pow(cp.DC, DCexp_);
      std::complex<float> const scale(dc * cp.weight, 0.f);
      cart.chip(icart[2], 3).chip(icart[1], 2).chip(icart[0], 1) +=
          radial.chip(iradial[1], 2).chip(iradial[0], 1) * scale;
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

  radial.setZero();
  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      auto const &iw = cp.cart;
      radial(iradial(0), iradial(1)) += cart(iw[0], iw[1], iw[2]) * cp.weight;
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
  auto grid_task = [&](long const lo_c, long const hi_c) {
    for (auto ic = lo_c; ic < hi_c; ic++) {
      auto const &cp = coords_[ic];
      auto const &iradial = cp.radial;
      auto const &iw = cp.cart;
      auto const &weight = radial.chip(iradial(1), 2).chip(iradial(0), 1).constant(cp.weight);
      radial.chip(iradial(1), 2).chip(iradial(0), 1) +=
          cart.chip(iw[2], 3).chip(iw[1], 2).chip(iw[0], 1) * weight;
    }
  };
  auto const &start = log_.start_time();
  Threads::RangeFor(grid_task, coords_.size());
  log_.stop_time(start, "Cartesian -> Radial");
}
