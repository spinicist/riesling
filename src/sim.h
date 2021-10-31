#pragma once

#include "types.h"

namespace Sim {

// A simple struct for returning multiple things from a simulation without tuple
struct Result
{
  Eigen::MatrixXf dynamics;
  Eigen::MatrixXf parameters;
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

  float value(long ii) const;

  private:
    float linspace(long i) const;
    float logspace(long i) const;
};

}
