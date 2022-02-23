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

Trajectory::Trajectory() {}

Trajectory::Trajectory(Info const &info, R3 const &points)
  : info_{info}
  , points_{points}

{
  echoes_ = I1(info_.spokes);
  echoes_.setZero();
  init();
}

Trajectory::Trajectory(Info const &info, R3 const &points, I1 const &echoes)
  : info_{info}
  , points_{points}
  , echoes_{echoes}

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
      Log::Fail(FMT_STRING("Maximum trajectory co-ordinate {} exceeded 0.5"), maxCoord);
    }
  } else {
    float const maxCoord = Maximum(
      points_.slice(Sz3{0, 0, 0}, Sz3{2, points_.dimension(1), points_.dimension(2)}).abs());
    if (maxCoord > 0.5f) {
      Log::Fail(FMT_STRING("Maximum in-plane trajectory {} co-ordinate exceeded 0.5"), maxCoord);
    }
  }

  Log::Print(FMT_STRING("Created trajectory object with {} spokes"), info_.spokes);
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
  assert(spoke < info_.spokes);

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

Mapping Trajectory::mapping(
  Index const kw, float const os, Index const inChan, float const inRes, bool const shrink) const
{
  Index const nChan = (inChan < 1) ? info_.channels : inChan;
  Index const kRad = kw / 2; // Radius to avoid at edge of grid
  float const res = inRes < 0.f ? info_.voxel_size.minCoeff() : inRes;
  float const ratio = info_.voxel_size.minCoeff() / res;
  Index const gridSz = fft_size(info_.matrix.maxCoeff() * os * (shrink ? ratio : 1.f));
  Log::Print(
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
  mapping.noncartDims = Sz3{nChan, info_.read_points, info_.spokes};
  mapping.scale = sqrt(info_.type == Info::Type::ThreeD ? pow(os, 3) : pow(os, 2));
  Index const totalSz = info_.read_points * info_.spokes;
  mapping.cart.reserve(totalSz);
  mapping.noncart.reserve(totalSz);
  mapping.echo.reserve(totalSz);
  mapping.offset.reserve(totalSz);
  mapping.echoes = Maximum(echoes_) + 1;
  mapping.echoWeights = Eigen::ArrayXf::Zero(mapping.echoes);
  std::fesetround(FE_TONEAREST);
  float const maxRad = ratio * ((gridSz / 2) - 1.f);
  Size3 const center(mapping.cartDims[0] / 2, mapping.cartDims[1] / 2, mapping.cartDims[2] / 2);
  auto start = Log::Now();
  for (int32_t is = 0; is < info_.spokes; is++) {
    auto const echo = echoes_(is);
    if ((echo >= 0) && (echo < info_.echoes)) {
      for (int16_t ir = 0; ir < info_.read_points; ir++) {
        NoncartesianIndex const nc{.spoke = is, .read = ir};
        Point3 const xyz = point(ir, is, maxRad);
        Point3 const gp = nearby(xyz);
        if (((gp.array().abs() + kRad) < maxRad).all()) {
          Size3 const cart = center + Size3(gp.cast<int16_t>());
          mapping.cart.push_back(CartesianIndex{cart(0), cart(1), cart(2)});
          mapping.noncart.push_back(nc);
          mapping.echo.push_back(echo);
          mapping.echoWeights[echo] += 1;
          mapping.offset.push_back(xyz - gp.cast<float>().matrix());
        }
      }
    }
  }
  Log::Print(
    FMT_STRING("Kept {} co-ords, {} discarded. Time {}"),
    mapping.cart.size(),
    totalSz - mapping.cart.size(),
    Log::ToNow(start));

  mapping.echoWeights = mapping.echoWeights.maxCoeff() / mapping.echoWeights;
  Log::Print(FMT_STRING("Echo weights: {}"), mapping.echoWeights.transpose());

  start = Log::Now();
  mapping.sortedIndices.resize(mapping.cart.size());
  std::iota(mapping.sortedIndices.begin(), mapping.sortedIndices.end(), 0);
  std::sort(
    mapping.sortedIndices.begin(), mapping.sortedIndices.end(), [&](Index const a, Index const b) {
      auto const &ac = mapping.cart[a];
      auto const &bc = mapping.cart[b];
      return (ac.z < bc.z) ||
             ((ac.z == bc.z) && ((ac.y < bc.y) || ((ac.y == bc.y) && (ac.x < bc.x))));
    });
  Log::Debug(FMT_STRING("Grid co-ord sorting: {}"), Log::ToNow(start));

  return mapping;
}

Trajectory Trajectory::downsample(float const dsIn, Cx4 &ks) const
{
  Info dsInfo = info_;
  // Account for rounding
  dsInfo.matrix = (info_.matrix.cast<float>() / dsIn).cast<Index>();
  float const ds = static_cast<float>(info_.matrix[0]) / dsInfo.matrix[0];
  dsInfo.voxel_size = info_.voxel_size * ds;
  Index sz = 3; // Need this for slicing below
  if (dsInfo.type == Info::Type::ThreeDStack) {
    dsInfo.matrix[2] = info_.matrix[2];
    dsInfo.voxel_size[2] = info_.voxel_size[2];
    sz = 2;
  }
  Index minRead = info_.read_points, maxRead = 0;
  R3 dsPoints(points_.dimensions());
  for (Index is = 0; is < info_.spokes; is++) {
    for (Index ir = 0; ir < info_.read_points; ir++) {
      R1 p = points_.chip<2>(is).chip<1>(ir);
      p.slice(Sz1{0}, Sz1{sz}) *= p.slice(Sz1{0}, Sz1{sz}).constant(ds);
      if (Norm(p.slice(Sz1{0}, Sz1{sz})) <= 0.5f) {
        dsPoints.chip<2>(is).chip<1>(ir) = p;
        minRead = std::min(minRead, ir);
        maxRead = std::max(maxRead, ir);
      } else {
        dsPoints.chip<2>(is).chip<1>(ir).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  dsInfo.read_points = maxRead - minRead;
  dsPoints = R3(dsPoints.slice(Sz3{0, minRead, 0}, Sz3{3, dsInfo.read_points, dsInfo.spokes}));
  ks = Cx4(ks.slice(
    Sz4{0, minRead, 0, 0},
    Sz4{dsInfo.channels, dsInfo.read_points, dsInfo.spokes, dsInfo.volumes}));
  Log::Print(
    FMT_STRING("Downsampled by {}, new voxel-size {} matrix {}, kept read-points {}-{}"),
    ds,
    dsInfo.voxel_size.transpose(),
    dsInfo.matrix.transpose(),
    minRead,
    maxRead);
  return Trajectory(dsInfo, dsPoints, echoes_);
}
