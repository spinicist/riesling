#pragma once

#include "types.h"

// A simple struct for returning multiple things from a simulation without tuple
struct SimResult
{
  Eigen::MatrixXf dynamics;
  Eigen::MatrixXf parameters;
};
