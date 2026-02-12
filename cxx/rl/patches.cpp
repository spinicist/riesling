#include "patches.hpp"
#include "log/log.hpp"
#include "sys/threads.hpp"

namespace rl {

template <int D> void Patches(Index const             patchSize,
                              Index const             windowSize,
                              bool const              doShift,
                              PatchFunction<D> const &apply,
                              CxNCMap<3 + D>          x,
                              CxNMap<3 + D>           y)
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
  Sz<3> const     p3{patchSize, patchSize, patchSize};
  Sz<3 + D> const szP = Concatenate(p3, LastN<D>(x.dimensions()));
  Index const     inset = (patchSize - windowSize) / 2;

  auto task = [&](Index const ilo, Index const istr) {
    for (Index iz = ilo; iz < nWindows[2]; iz += istr) {
      for (Index iy = 0; iy < nWindows[1]; iy++) {
        for (Index ix = 0; ix < nWindows[0]; ix++) {
          Sz3       ind{ix - 1, iy - 1, iz - 1};
          Sz<3 + D> stP, stW, stW2, szW; /* Will default initialize everything to zero thankfully */
          std::copy_n(y.dimensions().begin() + 3, D, szW.begin() + 3);
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
            CxN<3 + D> xp = x.slice(stP, szP);
            CxN<3 + D> yp(szP);
            apply(xp, yp);
            y.slice(stW, szW) = yp.slice(stW2, szW);
          }
        }
      }
    }
  };
  Threads::StridedFor(nWindows[2], task);
}

template void Patches<2>(Index const, Index const, bool const, PatchFunction<2> const &, CxNCMap<5>, CxNMap<5>);
template void Patches<3>(Index const, Index const, bool const, PatchFunction<3> const &, CxNCMap<6>, CxNMap<6>);

} // namespace rl
