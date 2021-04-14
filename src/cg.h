#pragma once

#include "log.h"
#include "types.h"

/* Conjugate gradients as described in K. P. Pruessmann, M. Weiger, M. B. Scheidegger, and P.
 * Boesiger, ‘SENSE: Sensitivity encoding for fast MRI’, Magnetic Resonance in Medicine, vol. 42,
 * no. 5, pp. 952–962, 1999.
 */
using CgSystem = std::function<void(Cx3 const &x, Cx3 &y)>;
void cg(CgSystem const &system, float const &thresh, long const &max_its, Cx3 &img, Log &log);

using CgVarSystem = std::function<void(Cx3 const &x, Cx3 &y, float const &pre)>;
void cgvar(
    CgVarSystem const &system,
    float const &thresh,
    long const &max_its,
    float const &pre0,
    float const &pre1,
    Cx3 &img,
    Log &log);
