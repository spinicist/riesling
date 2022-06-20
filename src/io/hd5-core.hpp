#pragma once

#include "hd5-keys.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace HD5 {

#include <hdf5.h>

using Handle = int64_t;

template <typename T>
struct type_tag
{
};

template <typename T>
hid_t type_impl(type_tag<T>);

template <typename T>
hid_t type()
{
  return type_impl(type_tag<T>{});
}

void Init();
Handle InfoType();
void CheckInfoType(Handle h);
bool Exists(Handle const h, std::string const name);
std::string GetError();
std::vector<std::string> List(Handle h);

} // namespace HD5
