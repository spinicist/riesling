#pragma once

#include "kernels.h"
#include "log.h"
#include "sdc.h"
#include "threads.h"
#include "trajectory.h"

#include <vector>

/** Transforms between non-cartesian and cartesian grids
 */
struct Gridder
{
  /** Constructs the Gridder with a default Cartesian grid
   *
   * @param traj  Trajectory object
   * @param os     Oversampling factor
   * @param kernel Interpolation kernel
   * @param unsafe Ignore thread safety
   * @param log    Logging object
   */
  Gridder(
      Trajectory const &traj, float const os, Kernel *const kernel, bool const unsafe, Log &log);

  virtual ~Gridder() = default;

  Dims3 gridDims() const; //!< Returns the dimensions of the grid
  Cx4 newGrid() const;    //!< Returns a correctly sized multi-channel grid
  Cx3 newGrid1() const;   //!< Returns a correctly sized single channel grid

  void setSDCExponent(float const dce); //!< Sets the exponent of the density compensation weights
  void setSDC(float const dc);
  void setSDC(R2 const &sdc);
  void setUnsafe();
  void setSafe();
  Info const &info() const;
  float oversample() const;
  Kernel *kernel() const;
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

  void genCoords(Trajectory const &traj, long const nomRad);
  void sortCoords();
  Info const info_;
  std::vector<Coords> coords_;
  std::vector<int32_t> sortedIndices_;
  Dims3 dims_;
  float oversample_, DCexp_, maxRad_;
  Kernel *kernel_;
  bool safe_;
  Log &log_;
};
