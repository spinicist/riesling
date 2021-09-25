#include "trajectory.h"

#include "tensorOps.h"
#include <cfenv>
#include <cmath>

// Helper function to convert a floating-point vector-like expression to integer values
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

Trajectory::Trajectory(Info const &info, R3 const &points, Log const &log)
    : info_{info}
    , points_{points}
    , log_{log}
{
  if (info_.read_points != points_.dimension(1)) {
    Log::Fail(
        "Mismatch between info read points {} and trajectory points {}",
        info_.read_points,
        points_.dimension(1));
  }
  if (info_.spokes_total() != points_.dimension(2)) {
    Log::Fail(
        "Mismatch between info spokes {} and trajectory spokes {}",
        info_.spokes_total(),
        points_.dimension(2));
  }

  if (info_.type == Info::Type::ThreeD) {
    float const maxCoord = Maximum(points_.abs());
    if (maxCoord > 0.5f) {
      Log::Fail("Maximum trajectory co-ordinate {} exceeded 0.5", maxCoord);
    }
  } else {
    float const maxCoord = Maximum(
        points_.slice(Sz3{0, 0, 0}, Sz3{2, points_.dimension(1), points_.dimension(2)}).abs());
    if (maxCoord > 0.5f) {
      Log::Fail("Maximum in-plane trajectory {} co-ordinate exceeded 0.5", maxCoord);
    }
  }

  Eigen::ArrayXf ind = Eigen::ArrayXf::LinSpaced(info_.read_points, 0, info_.read_points - 1);
  mergeHi_ = ind - (info_.read_gap - 1);
  mergeHi_ = (mergeHi_ > 0).select(mergeHi_, 0);
  mergeHi_ = (mergeHi_ < 1).select(mergeHi_, 1);

  if (info_.spokes_lo) {
    ind = Eigen::ArrayXf::LinSpaced(info_.read_points, 0, info_.read_points - 1);
    mergeLo_ = ind / info_.lo_scale - (info_.read_gap - 1);
    mergeLo_ = (mergeLo_ > 0).select(mergeLo_, 0);
    mergeLo_ = (mergeLo_ < 1).select(mergeLo_, 1);
    mergeLo_ = (1 - mergeLo_) / info_.lo_scale; // Match intensities of k-space
    mergeLo_.head(info_.read_gap) = 0.;         // Don't touch these points
  }
  log_.info("Created trajectory object with {} spokes", info_.spokes_total());
}

Info const &Trajectory::info() const
{
  return info_;
}

R3 const &Trajectory::points() const
{
  return points_;
}

Point3 Trajectory::point(int16_t const read, int32_t const spoke, float const rad_hi) const
{
  assert(read < info_.read_points);
  assert(spoke < info_.spokes_total());

  // Convention is to store the points between -0.5 and 0.5, so we need a factor of 2 here
  float const diameter = 2.f * (spoke < info_.spokes_lo ? rad_hi / info_.lo_scale : rad_hi);
  R1 const p = points_.chip(spoke, 2).chip(read, 1);
  switch (info_.type) {
  case Info::Type::ThreeD:
    return Point3{p(0) * diameter, p(1) * diameter, p(2) * diameter};
  case Info::Type::ThreeDStack:
    return Point3{p(0) * diameter, p(1) * diameter, p(2)};
  }
  __builtin_unreachable(); // Because the GCC devs are very obtuse
}

float Trajectory::merge(int16_t const read, int32_t const spoke) const
{
  if (spoke < info_.spokes_lo) {
    return mergeLo_(read);
  } else {
    return mergeHi_(read);
  }
}

Mapping
Trajectory::mapping(float const os, long const kRad, float const inRes, bool const shrink) const
{
  float const res = inRes < 0.f ? info_.voxel_size.minCoeff() : inRes;
  float const ratio = info_.voxel_size.minCoeff() / res;
  long const gridSz = fft_size(info_.matrix.maxCoeff() * os * (shrink ? ratio : 1.f));
  log_.info(
      FMT_STRING("Generating mapping to grid size {} at {} mm effective resolution"), gridSz, res);

  Mapping mapping;
  switch (info_.type) {
  case Info::Type::ThreeD:
    mapping.cartDims = Sz3{gridSz, gridSz, gridSz};
    break;
  case Info::Type::ThreeDStack:
    mapping.cartDims = Sz3{gridSz, gridSz, info_.matrix[2]};
    break;
  }
  mapping.osamp = os;
  long const totalSz = info_.read_points * info_.spokes_total();
  mapping.cart.reserve(totalSz);
  mapping.noncart.reserve(totalSz);
  mapping.sdc.reserve(totalSz);
  mapping.offset.reserve(totalSz);

  std::fesetround(FE_TONEAREST);
  float const maxRad = ratio * ((gridSz / 2) - 1.f);
  Size3 const center(mapping.cartDims[0] / 2, mapping.cartDims[1] / 2, mapping.cartDims[2] / 2);
  float const maxLoRad = maxRad * (float)(info_.read_gap) / (float)info_.read_points;
  float const maxHiRad = maxRad - kRad;
  auto start = log_.now();
  for (int32_t is = 0; is < info_.spokes_total(); is++) {
    for (int16_t ir = info_.read_gap; ir < info_.read_points; ir++) {
      NoncartesianIndex const nc{.spoke = is, .read = ir};
      Point3 const xyz = point(ir, is, maxRad);

      // Only grid lo-res to where hi-res begins (i.e. fill the dead-time gap)
      // Otherwise leave space for kernel
      if (xyz.norm() <= (is < info_.spokes_lo ? maxLoRad : maxHiRad)) {
        Size3 const gp = nearby(xyz).cast<int16_t>();
        Size3 const cart = gp + center;
        mapping.cart.push_back(CartesianIndex{cart(0), cart(1), cart(2)});
        mapping.noncart.push_back(nc);
        mapping.sdc.push_back(1.f);
        mapping.offset.push_back(xyz - gp.cast<float>().matrix());
      }
    }
  }
  log_.info("Generated {} co-ordinates in {}", mapping.cart.size(), log_.toNow(start));

  start = log_.now();
  mapping.sortedIndices.resize(mapping.cart.size());
  std::iota(mapping.sortedIndices.begin(), mapping.sortedIndices.end(), 0);
  std::sort(
      mapping.sortedIndices.begin(), mapping.sortedIndices.end(), [&](long const a, long const b) {
        auto const &ac = mapping.cart[a];
        auto const &bc = mapping.cart[b];
        return (ac.z < bc.z) ||
               ((ac.z == bc.z) && ((ac.y < bc.y) || ((ac.y == bc.y) && (ac.x < bc.x))));
      });
  log_.debug("Grid co-ord sorting: {}", log_.toNow(start));

  return mapping;
}
