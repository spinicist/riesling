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

void Array2fReader::operator()(std::string const &name, std::string const &value, Eigen::Array2f &v)
{
  if (auto result = scn::scan<float, float>(value, "{},{}")) {
    v[0] = std::get<0>(result->values());
    v[1] = std::get<1>(result->values());
  } else {
    throw(ArgsError(fmt::format("Could not read vector for {} from value {}", name, value)));
  }
}

void Array3fReader::operator()(std::string const &name, std::string const &value, Eigen::Array3f &v)
{
  if (auto result = scn::scan<float, float, float>(value, "{},{},{}")) {
    v[0] = std::get<0>(result->values());
    v[1] = std::get<1>(result->values());
    v[2] = std::get<2>(result->values());
  } else {
    throw(ArgsError(fmt::format("Could not read vector for {} from value {}", name, value)));
  }
}

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
