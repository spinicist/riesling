#include "parameter.hpp"

#include "log.hpp"
#include <random>

namespace rl {

auto ParameterGrid(Index const           nPar,
                   Eigen::ArrayXf const &lo,
                   Eigen::ArrayXf const &hi,
                   Eigen::ArrayXf const &delta) -> Eigen::ArrayXXf
{
  if (lo.size() != nPar) { Log::Fail("Low parameter size had {} elemeents, expected {}", lo.size(), nPar); }
  if (hi.size() != nPar) { Log::Fail("Low parameter size had {} elemeents, expected {}", hi.size(), nPar); }
  if (delta.size() != nPar) { Log::Fail("Low parameter size had {} elemeents, expected {}", delta.size(), nPar); }

  Eigen::ArrayXi N(nPar);
  Index          nTotal = 1;
  for (int ii = 0; ii < nPar; ii++) {
    N[ii] = (delta[ii] > 0.f) ? 1 + (hi[ii] - lo[ii]) / delta[ii] : 1;
    nTotal *= N[ii];
  }

  Eigen::ArrayXXf p(nPar, nTotal);
  Index           ind = 0;
  std::function<void(Index, Eigen::ArrayXf)> dimLoop = [&](Index dim, Eigen::ArrayXf pars) {
    for (Index id = 0; id < N[dim]; id++) {
      pars[dim] = lo[dim] + id * delta[dim];
      if (dim > 0) {
        dimLoop(dim - 1, pars);
      } else {
        p.col(ind++) = pars;
      }
    }
  };
  dimLoop(nPar - 1, Eigen::ArrayXf::Zero(nPar));
  return p;
}

} // namespace rl
