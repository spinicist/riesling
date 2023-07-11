#pragma once

#include "hd5-keys.hpp"
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
bool                     Exists(Handle const h, std::string const name);
void                     CheckedCall(int status, std::string const &msg);
std::string              GetError();
std::vector<std::string> List(Handle h);

} // namespace HD5
} // namespace rl
