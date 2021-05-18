#pragma once

#include "info.h"
#include "kernels.h"
#include "log.h"
#include "sdc.h"
#include "threads.h"

#include <vector>

/** Transforms between non-cartesian and cartesian grids
 */
struct Gridder
{
  /** Constructs the Gridder with a default Cartesian grid
   *
   * @param info   The Radial info struct
   * @param traj   The trajectory stored as X,Y,Z positions in k-space relative to k-max
   * @param os     Oversampling factor
   * @param kernel Interpolation kernel
   * @param fastgrid Ignore thread safety
   * @param log    Logging object
   * @param res    OPTIONAL - Desired effective resolution
   * @param shrink OPTIONAL - Shrink the grid to fit only the desired resolution portion
   */
  Gridder(
      Info const &info,
      R3 const &traj,
      float const os,
      Kernel *const kernel,
      bool const fastgrid,
      Log &log,
      float const res = 0.f,
      bool const shrink = false);

  virtual ~Gridder() = default;

  Dims3 gridDims() const; //!< Returns the dimensions of the grid
  Cx4 newGrid() const;    //!< Returns a correctly sized multi-channel grid
  Cx3 newGrid1() const;   //!< Returns a correctly sized single channel grid

  void setSDCExponent(float const dce); //!< Sets the exponent of the density compensation weights
  void setSDC(float const dc);
  void setSDC(R2 const &sdc);
  void setUnsafe();
  void setSafe();
  void toCartesian(Cx2 const &noncart, Cx3 &cart) const; //!< Single-channel non-cartesian -> cart
  void toCartesian(Cx3 const &noncart, Cx4 &cart) const; //!< Multi-channel non-cartesian -> cart
  void toNoncartesian(Cx3 const &cart, Cx2 &noncart) const; //!< Single-channel cart -> non-cart
  void toNoncartesian(Cx4 const &cart, Cx3 &noncart) const; //!< Multi-channel cart -> non-cart

protected:
  struct CartesianIndex
  {
    int16_t x, y, z;
  };

  struct NoncartesianIndex
  {
    int32_t spoke;
    int16_t read;
  };

  struct Coords
  {
    CartesianIndex cart;
    NoncartesianIndex noncart;
    float sdc;
    Point3 offset;
  };

  struct SpokeInfo_t
  {
    int16_t lo = std::numeric_limits<int16_t>::max();
    int16_t hi = std::numeric_limits<int16_t>::min();
    float xy = 0.f;
    float z = 0.f;
  };

  SpokeInfo_t spokeInfo(R2 const &traj, long const nomRad, float const maxRad, float const scale);
  std::vector<Coords>
  genCoords(R3 const &traj, int32_t const spoke0, long const spokes, SpokeInfo_t const &s);
  void sortCoords();
  Info const info_;
  std::vector<Coords> coords_;
  std::vector<int32_t> sortedIndices_;
  Dims3 dims_;
  float oversample_, DCexp_;
  Kernel *kernel_;
  bool safe_;
  Log &log_;
};
