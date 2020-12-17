#pragma once

#include "log.h"
#include "radial.h"
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
      RadialInfo const &info,
      R3 const &traj,
      float const os,
      bool const est_dc,
      bool const kb,
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
  void toCartesian(Cx2 const &radial, Cx3 &cart) const; //!< Single-channel non-cartesian -> cart
  void toCartesian(Cx3 const &radial, Cx4 &cart) const; //!< Multi-channel non-cartesian -> cart
  void toRadial(Cx3 const &cart, Cx2 &radial) const;    //!< Single-channel cart -> non-cart
  void toRadial(Cx4 const &cart, Cx3 &radial) const;    //!< Multi-channel cart -> non-cart

  void apodize(Cx3 &img) const;   //!< Apodize interpolation kernel
  void deapodize(Cx3 &img) const; //!< De-apodize interpolation kernel

private:
  struct Coords
  {
    Size3 cart;
    Size2 radial;
    float DC;
    float weight;
  };

  std::vector<Coords> stackCoords(
      Eigen::TensorMap<R3 const> const &traj,
      long const spokeOffset,
      long const nomRad,
      float const maxRad,
      float const scale);
  std::vector<Coords> fullCoords(
      Eigen::TensorMap<R3 const> const &traj,
      long const spokeOffset,
      long const nomRad,
      float const maxRad,
      float const scale);
  void sortCoords();
  void iterativeDC(); //!< Iteratively estimate the density-compensation weights
  void stackKernel(R3 const &traj, long const nomRad);
  void kbApodization(bool const stack);
  RadialInfo const info_;
  std::vector<Coords> coords_;
  std::vector<long> sortedIndices_;
  Dims3 dims_;
  float oversample_, DCexp_, kbBeta_;
  long kbW_;
  Eigen::ArrayXf apodX_, apodY_, apodZ_;
  Log &log_;
};
