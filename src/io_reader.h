#pragma once

#include "log.h"
#include "radial.h"
#include <map>

/** Defines the interface for reading vendor data files, e.g. P-Files or ScanArchives
 *
 */
struct Reader
{
  virtual ~Reader(){};

  virtual std::string ID() const = 0;     //< Returns the ID/suggested filename
  virtual RadialInfo getInfo() const = 0; //< Returns the radial header/information struct
  virtual std::map<std::string, float> getMeta() const = 0; //< Returns the meta-data from this file
  /** Returns the trajectory for this scan
   *
   * The trajectory is stored as a 3D tensor of size (4, Nr, Ns) where Nr is the number of read-out
   * points and Ns is the number of spokes. The 4 elements of each stored point are X,Y,Z,C where C
   * is the density compensation factor and X,Y,Z are the position of the sample in Cartesian
   * k-space, where 1.0 is the k-space extremity.
   */
  virtual R3 getTrajectory(bool nolo = false) const = 0;
  virtual void readNextVolume(long const iv, Cx4 &data) = 0; //< Reads next volume in the file
};
