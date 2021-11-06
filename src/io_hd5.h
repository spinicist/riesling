#pragma once

#include <cstdint>
#include <string>

namespace HD5 {
using Handle = int64_t;

void Init();
Handle InfoType();
void CheckInfoType(Handle h);

namespace Keys {
std::string const Info = "info";
std::string const Meta = "meta";
std::string const Noncartesian = "noncartesian";
std::string const Cartesian = "cartesian";
std::string const Image = "image";
std::string const Trajectory = "trajectory";
std::string const Basis = "basis";
std::string const BasisImages = "basis-images";
std::string const Dynamics = "dynamics";
std::string const SDC = "sdc";
std::string const SENSE = "sense";
} // namespace Keys

} // namespace HD5
