#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "grid-base.hpp"
#include <memory>

std::unique_ptr<GridBase>
make_grid(Kernel const *k, Mapping const &m, Index const nC, std::string const &basis = "");
