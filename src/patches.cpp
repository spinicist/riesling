#include "patches.hpp"
#include "log.hpp"
#include "threads.hpp"

namespace rl {

void Patches(
  Index const                        patchSize,
  Index const                        windowSize,
  bool const                         doShift,
  PatchFunction const               &apply,
  Eigen::TensorMap<Cx4 const> const &x,
  Eigen::TensorMap<Cx4>             &y)
{
  Sz3 nWindows, shift;

  for (Index ii = 0; ii < 3; ii++) {
    auto const d = x.dimension(ii + 1);
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

  Log::Print<Log::Level::Debug>("Windows {} Shifts {}", nWindows, shift);
  Sz4 const   szP{x.dimension(0), patchSize, patchSize, patchSize};
  Index const inset = (patchSize - windowSize) / 2;

  auto zTask = [&](Index const iz) {
    for (Index iy = 0; iy < nWindows[1]; iy++) {
      for (Index ix = 0; ix < nWindows[0]; ix++) {
        Sz3 ind{ix - 1, iy - 1, iz - 1};
        Sz4 stP, stW, stW2, szW;
        stP[0] = stW[0] = stW2[0] = 0;
        szW[0] = y.dimension(0);
        bool empty = false;
        for (Index ii = 0; ii < 3; ii++) {
          Index const d = x.dimension(ii + 1);
          Index const st = ind[ii] * windowSize + shift[ii];
          stW[ii + 1] = std::max(st, 0L);
          szW[ii + 1] = windowSize + std::min({st, 0L, d - stW[ii + 1] - windowSize});
          if (szW[ii + 1] < 1) {
            empty = true;
            break;
          }
          stP[ii + 1] = std::clamp(st - inset, 0L, d - patchSize);
          stW2[ii + 1] = stW[ii + 1] - stP[ii + 1];
        }
        if (empty) { continue; }
        Cx4 xp = x.slice(stP, szP);
        Cx4 yp = apply(xp);
        y.slice(stW, szW) = yp.slice(stW2, szW);
      }
    }
  };
  Threads::For(zTask, nWindows[2], "Patches");
}

} // namespace rl
