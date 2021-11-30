#pragma once

#include "log.h"
#include "types.h"

namespace Sim {

// A simple struct for returning multiple things from a simulation without tuple
struct Result
{
  Eigen::ArrayXXf dynamics;
  Eigen::ArrayXXf parameters;
  Eigen::ArrayXf Mz_ss;
};

// Another simple struct for passing around the main ZTE sequence parameters
struct Sequence
{
  Index sps;
  float alpha, TR, Tramp, Tssi, TI, Trec, TE;
};

// Arg lists are getting annoyingly Index
struct Parameter
{
  Index N;
  float lo, hi;
  bool logspaced;

  float value(Index const ii) const;

private:
  float linspace(Index const i) const;
  float logspace(Index const i) const;
};

template <int NP>
struct ParameterGenerator
{
  using Parameters = Eigen::Array<float, NP, 1>;

  ParameterGenerator(std::array<Parameter, NP> const &p)
    : pars{p}
  {
  }

  Index totalN() const
  {
    Index total = 1;
    for (auto const &par : pars) {
      total *= par.N;
    }
    return total;
  }

  Parameters values(Index const ii)
  {
    assert(ii < totalN());
    Parameters p;
    Index next = ii;
    for (Index ip = 0; ip < NP; ip++) {
      auto result = std::div(next, pars[ip].N);
      p(ip) = pars[ip].value(result.rem);
      next = result.quot;
    }
    return p;
  }

  Parameters rand() const
  {
    Parameters p;
    // Eigen random range is -1 to 1, scale appropriately
    p.setRandom();
    for (Index ip = 0; ip < NP; ip++) {
      p(ip) = pars[ip].lo + (p(ip) + 1.f) * (pars[ip].hi - pars[ip].lo) / 2.f;
    }
    return p;
  }

private:
  std::array<Parameter, NP> pars;
};

} // namespace Sim
