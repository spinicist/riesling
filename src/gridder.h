#pragma once

#include "info.h"
#include "kernels.h"
#include "log.h"
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
   * @param est_dc Estimate DC using Menon & Pipe iterative method
   * @param kb     Use Kaiser-Bessel interpolation
   * @param stack  Trajectory is stack-of-stars or similar
   * @param log    Logging object
   * @param res    OPTIONAL - Desired effective resolution
   * @param shrink OPTIONAL - Shrink the grid to fit only the desired resolution portion
   */
  Gridder(
      Info const &info,
      R3 const &traj,
      float const os,
      bool const est_dc,
      Kernel *const kernel,
      bool const stack,
      Log &log,
      float const res = 0.f,
      bool const shrink = false);

  virtual ~Gridder() = default;

  Dims3 gridDims() const; //!< Returns the dimensions of the grid
  Cx4 newGrid() const;    //!< Returns a correctly sized multi-channel grid
  Cx3 newGrid1() const;   //!< Returns a correctly sized single channel grid

  void setDCExponent(float const dce); //!< Sets the exponent of the density compensation weights
  void setDC(float const dc);
  void toCartesian(Cx2 const &noncart, Cx3 &cart) const; //!< Single-channel non-cartesian -> cart
  void toCartesian(Cx3 const &noncart, Cx4 &cart) const; //!< Multi-channel non-cartesian -> cart
  void toNoncartesian(Cx3 const &cart, Cx2 &noncart) const; //!< Single-channel cart -> non-cart
  void toNoncartesian(Cx4 const &cart, Cx3 &noncart) const; //!< Multi-channel cart -> non-cart

  void apodize(Cx3 &img) const;   //!< Apodize interpolation kernel
  void deapodize(Cx3 &img) const; //!< De-apodize interpolation kernel

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
    float DC;
    Point3 offset;
  };

  struct Profile
  {
    int16_t lo = std::numeric_limits<int16_t>::max();
    int16_t hi = std::numeric_limits<int16_t>::min();
    float xy;
    float z;
    std::vector<float> DC;
  };

  Profile profile2D(
      R2 const &traj, long const spokes, long const nomRad, float const maxRad, float const scale);
  Profile profile3D(
      R2 const &traj, long const spokes, long const nomRad, float const maxRad, float const scale);

  std::vector<Coords>
  genCoords(R3 const &traj, int32_t const spoke0, long const spokes, Profile const &profile);
  void sortCoords();
  void iterativeDC(); //!< Iteratively estimate the density-compensation weights
  Info const info_;
  std::vector<Coords> coords_;
  std::vector<int32_t> sortedIndices_;
  Dims3 dims_;
  float oversample_, DCexp_;
  Kernel *kernel_;
  ApodizeFunction apodize_;
  Log &log_;
};
