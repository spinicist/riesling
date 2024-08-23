#include "parameter.hpp"

#include "log.hpp"
#include <random>

namespace rl {

auto ParameterGrid(Index const nPar, Eigen::ArrayXf const &lo, Eigen::ArrayXf const &hi, Eigen::ArrayXi const &N)
  -> Eigen::ArrayXXf
{
  if (lo.size() != nPar) { Log::Fail("Parameter low values had {} elements, expected {}", lo.size(), nPar); }
  if (hi.size() != nPar) { Log::Fail("Parameter high values had {} elements, expected {}", hi.size(), nPar); }
  if (N.size() != nPar) { Log::Fail("Parameter N had {} elements, expected {}", N.size(), nPar); }

  Eigen::ArrayXf delta(nPar);
  Index          nTotal = 1;
  for (int ii = 0; ii < nPar; ii++) {
    if (N[ii] < 1) {
      Log::Fail("Parameter {} N was less than 1", ii);
    } else if (N[ii] == 1) {
      delta[ii] = 0.f;
    } else {
      delta[ii] = (hi[ii] - lo[ii]) / (N[ii] - 1);
    }
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
