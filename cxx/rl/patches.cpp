#include "patches.hpp"
#include "log/log.hpp"
#include "sys/threads.hpp"

namespace rl {

void Patches(Index const patchSize, Index const windowSize, bool const doShift, PatchFunction const &apply, Cx5CMap x, Cx5Map y)
{
  if (x.dimensions() != y.dimensions()) {
    throw(Log::Failure("Patches", "Input shape {}, output shape {}", x.dimensions(), y.dimensions()));
  }
  Sz3 nWindows, shift;

  for (Index ii = 0; ii < 3; ii++) {
    auto const d = x.dimension(ii);
    nWindows[ii] = (d / windowSize) + 2;
  }

  if (doShift) {
    std::random_device rd;
    std::mt19937       gen(rd());
    for (Index ii = 0; ii < 3; ii++) {
      std::uniform_int_distribution<> int_dist(0, windowSize - 1);
      shift[ii] = int_dist(gen);
    }
  } else {
    for (Index ii = 0; ii < 3; ii++) {
      shift[ii] = 0;
    }
  }

  Log::Debug("Patch", "Windows {} Shifts {}", nWindows, shift);
  Sz5 const   szP{patchSize, patchSize, patchSize, x.dimension(3), x.dimension(4)};
  Index const inset = (patchSize - windowSize) / 2;

  auto task = [&](Index const ilo, Index const istr) {
    for (Index iz = ilo; iz < nWindows[2]; iz += istr) {
      for (Index iy = 0; iy < nWindows[1]; iy++) {
        for (Index ix = 0; ix < nWindows[0]; ix++) {
          Sz3 ind{ix - 1, iy - 1, iz - 1};
          Sz5 stP, stW, stW2, szW;
          stP[3] = stW[3] = stW2[3] = 0;
          stP[4] = stW[4] = stW2[4] = 0;
          szW[3] = y.dimension(3);
          szW[4] = y.dimension(4);
          bool empty = false;
          for (Index ii = 0; ii < 3; ii++) {
            Index const d = x.dimension(ii);
            Index const st = ind[ii] * windowSize + shift[ii];
            stW[ii] = std::max(st, 0L);
            szW[ii] = windowSize + std::min({st, 0L, d - stW[ii] - windowSize});
            if (szW[ii] < 1) {
              empty = true;
              break;
            }
            stP[ii] = std::clamp(st - inset, 0L, d - patchSize);
            stW2[ii] = stW[ii] - stP[ii];
          }
          if (!empty) {
            Cx5 xp = x.slice(stP, szP);
            Cx5 yp = apply(xp);
            y.slice(stW, szW) = yp.slice(stW2, szW);
          }
        }
      }
    }
  };
  Threads::StridedFor(nWindows[2], task);
}

} // namespace rl
