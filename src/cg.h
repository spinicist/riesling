#pragma once

#include "log.h"
#include "types.h"

/* Conjugate gradients as described in K. P. Pruessmann, M. Weiger, M. B. Scheidegger, and P.
 * Boesiger, ‘SENSE: Sensitivity encoding for fast MRI’, Magnetic Resonance in Medicine, vol. 42,
 * no. 5, pp. 952–962, 1999.
 */
void cg(SystemFunction const &system, long const &max_its, float const &thresh, Cx3 &img, Log &log);
