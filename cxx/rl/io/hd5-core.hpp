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
Handle type_impl(type_tag<T>, bool const alt = false);

template <typename T>
Handle type(bool const alt = false)
{
  return type_impl(type_tag<T>{}, alt);
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
std::string const Pars = "parameters";
std::string const ProtonDensity = "pd";
std::string const Residual = "residual";
std::string const Trajectory = "trajectory";
std::string const Weights = "weights";
} // namespace Keys

// Horrible hack due to DSizes shenanigans
template <int N>
struct DimensionNames : std::array<std::string, N> {};

namespace Dims {
DimensionNames<3> const Basis = {"b", "sample", "trace"};
DimensionNames<6> const Channels = {"b", "channel", "i", "j", "k", "t"};
DimensionNames<5> const Image = {"b", "i", "j", "k", "t"};
DimensionNames<5> const Noncartesian = {"channel", "sample", "trace", "slab", "t"};
DimensionNames<5> const SENSE = {"b", "channel", "i", "j", "k"};
DimensionNames<3> const Trajectory = {"k", "sample", "trace"};
} // namespace Dims

} // namespace HD5
} // namespace rl
