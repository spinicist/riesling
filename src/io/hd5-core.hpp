#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rl {
namespace HD5 {

using Handle = int64_t;

template <typename T>
struct type_tag
{
};

template <typename T>
Handle type_impl(type_tag<T>);

template <typename T>
Handle type()
{
  return type_impl(type_tag<T>{});
}

void                     Init();
Handle                   InfoType();
void                     CheckInfoType(Handle h);
auto                     Exists(Handle const h, std::string const &name) -> bool;
void                     CheckedCall(int status, std::string const &msg);
std::string              GetError();
std::vector<std::string> List(Handle h);

namespace Keys {
std::string const Basis = "basis";
std::string const CompressionMatrix = "ccmat";
std::string const Data = "data";
std::string const Dictionary = "dictionary";
std::string const Dynamics = "dynamics";
std::string const Info = "info";
std::string const Meta = "meta";
std::string const Norm = "norm";
std::string const Parameters = "parameters";
std::string const ProtonDensity = "pd";
std::string const ResidualImage = "resid-image";
std::string const ResidualKSpace = "resid-data";
std::string const Trajectory = "trajectory";
std::string const Weights = "weights";
} // namespace Keys

// Horrible hack due to DSizes shenanigans
template <int N>
struct Names : std::array<std::string, N> {};

namespace Dims {
Names<3> const Basis = {"v", "sample", "trace"};
Names<6> const Cartesian = {"channel", "v", "x", "y", "z", "t"};
Names<5> const Image = {"v", "x", "y", "z", "t"};
Names<5> const Noncartesian = {"channel", "sample", "trace", "slab", "t"};
Names<5> const SENSE = {"channel", "v", "x", "y", "z"};
Names<3> const Trajectory = {"k", "sample", "trace"};
} // namespace Dims

} // namespace HD5
} // namespace rl
