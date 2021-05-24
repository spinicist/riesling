#pragma once

#include "log.h"
#include "types.h"

/*!
 * Creates a virtual body-coil via SVD and normalizes the sensitivities to it
 */
void VBC(Cx4 &data, Log &log);

/*!
 * Centers the phase of a set of volumes by applying the Virtual Conjugate-Coil method
 */
void VCC(Cx4 &data, Log &log);
