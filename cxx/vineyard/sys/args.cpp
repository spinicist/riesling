#include "sys/args.hpp"

#include <exception>
#include <fmt/format.h>
#include <scn/scan.h>

class ArgsError : public std::runtime_error
{
public:
  ArgsError(std::string const &msg)
    : std::runtime_error(msg)
  {
  }
};

template <int N> void SzReader<N>::operator()(std::string const &name, std::string const &value, rl::Sz<N> &sz)
{
  size_t ind = 0;
  if (auto result = scn::scan<Index>(value, "{}")) {
    sz[ind] = result->value();
    for (ind = 1; ind < N; ind++) {
      result = scn::scan<Index>(result->range(), ",{}");
      if (!result) { throw(ArgsError(fmt::format("Could not read {} from '{}'", name, value))); }
      sz[ind] = result->value();
    }
  } else {
    throw(ArgsError(fmt::format("Could not read {} from '{}'", name, value)));
  }
}

template struct SzReader<2>;
template struct SzReader<3>;
template struct SzReader<4>;

template <typename T, int ND>
void ArrayReader<T, ND>::operator()(std::string const &name, std::string const &value, Eigen::Array<T, ND, 1> &v)
{
  size_t ind = 0;
  if (auto result = scn::scan<T>(value, "{}")) {
    v[ind] = result->value();
    for (ind = 1; ind < ND; ind++) {
      result = scn::scan<T>(result->range(), ",{}");
      if (!result) { throw(ArgsError(fmt::format("Could not read {} from '{}'", name, value))); }
      v[ind] = result->value();
    }
  } else {
    throw(ArgsError(fmt::format("Could not read {} from '{}'", name, value)));
  }
}

template struct ArrayReader<float, 1>;
template struct ArrayReader<float, 2>;
template struct ArrayReader<float, 3>;

void Vector3fReader::operator()(std::string const &name, std::string const &value, Eigen::Vector3f &v)
{
  if (auto result = scn::scan<float, float, float>(value, "{},{},{}")) {
    v[0] = std::get<0>(result->values());
    v[1] = std::get<1>(result->values());
    v[2] = std::get<2>(result->values());
  } else {
    throw(ArgsError(fmt::format("Could not read vector for {} from value {}", name, value)));
  }
}

void ArrayXfReader::operator()(std::string const &name, std::string const &input, Eigen::ArrayXf &val)
{
  auto               result = scn::scan<float>(input, "{}");
  std::vector<float> values;
  if (result) {
    values.push_back(result->value());
    while ((result = scn::scan<float>(result->range(), ",{}"))) {
      values.push_back(result->value());
    }
  } else {
    throw(ArgsError(fmt::format("Could not read argument for {}", name)));
  }
  val.resize(values.size());
  for (size_t ii = 0; ii < values.size(); ii++) {
    val[ii] = values[ii];
  }
}

void ArrayXiReader::operator()(std::string const &name, std::string const &input, Eigen::ArrayXi &val)
{
  auto             result = scn::scan<int>(input, "{}");
  std::vector<int> values;
  if (result) {
    values.push_back(result->value());
    while ((result = scn::scan<int>(result->range(), ",{}"))) {
      values.push_back(result->value());
    }
  } else {
    throw(ArgsError(fmt::format("Could not read argument for {}", name)));
  }
  val.resize(values.size());
  for (size_t ii = 0; ii < values.size(); ii++) {
    val[ii] = values[ii];
  }
}

template <typename T>
void VectorReader<T>::operator()(std::string const &name, std::string const &input, std::vector<T> &values)
{
  auto result = scn::scan<T>(input, "{}");
  if (result) {
    // Values will have been default initialized. Reset
    values.clear();
    values.push_back(result->value());
    while ((result = scn::scan<T>(result->range(), ",{}"))) {
      values.push_back(result->value());
    }
  } else {
    throw(ArgsError(fmt::format("Could not read argument for {}", name)));
  }
}

template struct VectorReader<float>;
template struct VectorReader<Index>;
