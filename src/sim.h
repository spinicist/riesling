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
  long sps;
  float alpha, TR, Tramp, Tssi, TI, Trec;
};

// Arg lists are getting annoyingly long
struct Parameter
{
  long N;
  float lo, hi;
  bool logspaced;

  float value(long const ii) const;

private:
  float linspace(long const i) const;
  float logspace(long const i) const;
};

template <int NP>
struct ParameterGenerator
{
  using Parameters = Eigen::Array<float, NP, 1>;

  ParameterGenerator(std::array<Parameter, NP> const &p)
      : pars{p}
  {
  }

  long totalN() const
  {
    long total = 1;
    for (auto const &par : pars) {
      total *= par.N;
    }
    return total;
  }

  Parameters values(long const ii)
  {
    assert(ii < totalN());
    Parameters p;
    long next = ii;
    for (long ip = 0; ip < NP; ip++) {
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
    for (long ip = 0; ip < NP; ip++) {
      p(ip) = pars[ip].lo + (p(ip) + 1.f) * (pars[ip].hi - pars[ip].lo) / 2.f;
    }
    return p;
  }

private:
  std::array<Parameter, NP> pars;
};

} // namespace Sim
