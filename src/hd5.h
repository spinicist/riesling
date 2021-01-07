#pragma once

#include "info.h"
#include <map>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

namespace HD5 {

using Handle = int64_t;
enum class Mode
{
  ReadOnly,
  WriteOnly
};

void init();

Handle open_file(std::string const &path, Mode const mode);
void close_file(Handle const &file);

std::string get_name(Handle const &file);

Handle create_group(Handle const &parent, std::string const &name);
Handle open_group(Handle const &parent, std::string const &name);
Handle open_data(Handle const &parent, std::string const &name);
void close_group(Handle const &group);
std::vector<std::string> list_group(Handle const &group);

template <typename T>
T load_data(Handle const &dset);

template <typename T, int N>
extern void load_tensor(Handle const &parent, std::string const &name, Eigen::Tensor<T, N> &tensor);
template <typename T, int N>
extern void
store_tensor(Handle const &parent, std::string const &name, Eigen::Tensor<T, N> const &data);

template <typename T, int R, int C>
extern void load_array(Handle const &parent, std::string const &name, Eigen::Array<T, R, C> &array);
template <typename T, int R, int C>
extern void
store_array(Handle const &parent, std::string const &name, Eigen::Array<T, R, C> const &array);

void store_info(Handle const &parent, Info const &info);
void load_info(Handle const &dset, Info &info);
void store_map(Handle const &parent, std::map<std::string, float> const &meta);
void load_map(Handle const &parent, std::map<std::string, float> &meta);

} // namespace HD5