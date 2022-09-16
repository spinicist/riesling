#pragma once

#include "log.hpp"
#include "types.hpp"

/*!
 * Creates a virtual body-coil via SVD
 */
Cx4 VBC(Cx4 &data);

/*!
 * Centers the phase of a set of volumes by applying the Virtual Conjugate-Coil method
 */
void VCC(Cx4 &data);

/*!
 * Creates a reference image via the Hammond method
 */
Cx3 Hammond(Cx4 const &data);
