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
inline Index fft_size(float const x)
{
  if (x > 8.f) {
    return (std::lrint(x) + 7L) & ~7L;
  } else {
    return (Index)std::ceil(x);
  }
}

Trajectory::Trajectory(Info const &info, R3 const &points, Log const &log)
  : info_{info}
  , points_{points}
  , log_{log}
{
  echoes_ = I1(info_.spokes);
  echoes_.setZero();
  init();
}

Trajectory::Trajectory(Info const &info, R3 const &points, I1 const &echoes, Log const &log)
  : info_{info}
  , points_{points}
  , echoes_{echoes}
  , log_{log}
{
  init();
}

void Trajectory::init()
{
  if (info_.read_points != points_.dimension(1)) {
    Log::Fail(
      "Mismatch between info read points {} and trajectory points {}",
      info_.read_points,
      points_.dimension(1));
  }
  if (info_.spokes != points_.dimension(2)) {
    Log::Fail(
      "Mismatch between info spokes {} and trajectory spokes {}",
      info_.spokes,
      points_.dimension(2));
  }
  if (info_.spokes != echoes_.dimension(0)) {
    Log::Fail(
      "Mismatch between info spokes {} and echoes array {}", info_.spokes, echoes_.dimension(0));
  }
  if (info_.echoes < Maximum(echoes_)) {
    Log::Fail(
      "Maximum echo {} exceeds number of echoes in header {}", Maximum(echoes_), info_.echoes);
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

  log_.info("Created trajectory object with {} spokes", info_.spokes);
}

Info const &Trajectory::info() const
{
  return info_;
}

R3 const &Trajectory::points() const
{
  return points_;
}

I1 const &Trajectory::echoes() const
{
  return echoes_;
}

Point3 Trajectory::point(int16_t const read, int32_t const spoke, float const rad_hi) const
{
  assert(read < info_.read_points);
  assert(spoke < info_.spokes_total());

  // Convention is to store the points between -0.5 and 0.5, so we need a factor of 2 here
  float const diameter = 2.f * rad_hi;
  R1 const p = points_.chip(spoke, 2).chip(read, 1);
  switch (info_.type) {
  case Info::Type::ThreeD:
    return Point3{p(0) * diameter, p(1) * diameter, p(2) * diameter};
  case Info::Type::ThreeDStack:
    return Point3{p(0) * diameter, p(1) * diameter, p(2)};
  }
  __builtin_unreachable(); // Because the GCC devs are very obtuse
}

Mapping
Trajectory::mapping(float const os, Index const kRad, float const inRes, bool const shrink) const
{
  float const res = inRes < 0.f ? info_.voxel_size.minCoeff() : inRes;
  float const ratio = info_.voxel_size.minCoeff() / res;
  Index const gridSz = fft_size(info_.matrix.maxCoeff() * os * (shrink ? ratio : 1.f));
  log_.info(
    FMT_STRING("Generating mapping to grid size {} at {} mm effective resolution"), gridSz, res);

  Mapping mapping;
  mapping.type = info_.type;
  switch (mapping.type) {
  case Info::Type::ThreeD:
    mapping.cartDims = Sz3{gridSz, gridSz, gridSz};
    break;
  case Info::Type::ThreeDStack:
    mapping.cartDims = Sz3{gridSz, gridSz, info_.matrix[2]};
    break;
  }
  mapping.noncartDims = Sz3{info_.channels, info_.read_points, info_.spokes};
  mapping.osamp = os;
  Index const totalSz = info_.read_points * info_.spokes;
  mapping.cart.reserve(totalSz);
  mapping.noncart.reserve(totalSz);
  mapping.echo.reserve(totalSz);
  mapping.sdc.reserve(totalSz);
  mapping.offset.reserve(totalSz);
  mapping.echoes = info_.echoes;
  std::fesetround(FE_TONEAREST);
  float const maxRad = ratio * ((gridSz / 2) - 1.f);
  Size3 const center(mapping.cartDims[0] / 2, mapping.cartDims[1] / 2, mapping.cartDims[2] / 2);
  auto start = log_.now();
  for (int32_t is = 0; is < info_.spokes; is++) {
    auto const echo = echoes_(is);
    if ((echo >= 0) && (echo < info_.echoes)) {
      for (int16_t ir = info_.read_gap; ir < info_.read_points; ir++) {
        NoncartesianIndex const nc{.spoke = is, .read = ir};
        Point3 const xyz = point(ir, is, maxRad);
        Point3 const gp = nearby(xyz);
        if (((gp.array().abs() + kRad) < maxRad).all()) {
          Size3 const cart = center + Size3(gp.cast<int16_t>());
          mapping.cart.push_back(CartesianIndex{cart(0), cart(1), cart(2)});
          mapping.noncart.push_back(nc);
          mapping.echo.push_back(echo);
          mapping.sdc.push_back(1.f);
          mapping.offset.push_back(xyz - gp.cast<float>().matrix());
        }
      }
    }
  }
  log_.info("Generated {} co-ordinates in {}", mapping.cart.size(), log_.toNow(start));

  start = log_.now();
  mapping.sortedIndices.resize(mapping.cart.size());
  std::iota(mapping.sortedIndices.begin(), mapping.sortedIndices.end(), 0);
  std::sort(
    mapping.sortedIndices.begin(), mapping.sortedIndices.end(), [&](Index const a, Index const b) {
      auto const &ac = mapping.cart[a];
      auto const &bc = mapping.cart[b];
      return (ac.z < bc.z) ||
             ((ac.z == bc.z) && ((ac.y < bc.y) || ((ac.y == bc.y) && (ac.x < bc.x))));
    });
  log_.debug("Grid co-ord sorting: {}", log_.toNow(start));

  return mapping;
}
