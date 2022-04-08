#pragma once

#include <cstdint>
#include <string>

namespace HD5 {
using Handle = int64_t;

void Init();
Handle InfoType();
void CheckInfoType(Handle h);
bool Exists(Handle const h, std::string const name);
std::string GetError();

namespace Keys {
std::string const Basis = "basis";
std::string const Cartesian = "cartesian";
std::string const Channels = "channels";
std::string const CompressionMatrix = "ccmat";
std::string const Dictionary = "dictionary";
std::string const Dynamics = "dynamics";
std::string const Image = "image";
std::string const Info = "info";
std::string const Meta = "meta";
std::string const Noncartesian = "noncartesian";
std::string const Norm = "norm";
std::string const Parameters = "parameters";
std::string const ProtonDensity = "pd";
std::string const SDC = "sdc";
std::string const SENSE = "sense";
std::string const Trajectory = "trajectory";
} // namespace Keys

} // namespace HD5