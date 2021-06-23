#pragma once

#include "log.h"
#include "types.h"

/*!
 * Creates a virtual body-coil via SVD
 */
Cx4 VBC(Cx4 &data, Log &log);

/*!
 * Centers the phase of a set of volumes by applying the Virtual Conjugate-Coil method
 */
void VCC(Cx4 &data, Log &log);

/*!
 * Creates a reference image via the Hammond method
 */
Cx3 Hammond(Cx4 const &data, Log &log);
