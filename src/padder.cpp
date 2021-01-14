#include "padder.h"

void ZeroPad(Cx4 const &small, Cx4 &large)
{
  auto const largeSz = large.dimensions();
  auto const smallSz = small.dimensions();

  large.setZero();
  large.slice(
      Sz4{0,
          (largeSz[1] - smallSz[1]) / 2,
          (largeSz[2] - smallSz[2]) / 2,
          (largeSz[3] - smallSz[3]) / 2},
      smallSz) = small;
}
