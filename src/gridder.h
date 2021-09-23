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
   * @param map    Mapping between non-Cartesian and Cartesian
   * @param kernel Interpolation kernel
   * @param unsafe Ignore thread safety
   * @param log    Logging object
   */
  Gridder(Mapping map, Kernel *const kernel, bool const unsafe, Log &log);

  virtual ~Gridder() = default;

  Sz3 gridDims() const;                        //!< Returns the dimensions of the grid
  Cx4 newMultichannel(long const nChan) const; //!< Returns a correctly sized multi-channel grid

  void setSDCExponent(float const dce); //!< Sets the exponent of the density compensation weights
  void setSDC(float const dc);
  void setSDC(R2 const &sdc);
  void setUnsafe();
  void setSafe();
  Kernel *kernel() const;
  void toCartesian(Cx3 const &noncart, Cx4 &cart) const;    //!< Multi-channel non-cartesian -> cart
  void toNoncartesian(Cx4 const &cart, Cx3 &noncart) const; //!< Multi-channel cart -> non-cart

protected:
  Mapping mapping_;
  Kernel *kernel_;
  bool safe_;
  Log &log_;
  float DCexp_;
};
