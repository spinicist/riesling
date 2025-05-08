#include "debug.hpp"

#include "../io/hd5.hpp"

namespace rl {
namespace Log {

namespace {
std::shared_ptr<HD5::Writer> debug_file = nullptr;
}

void SetDebugFile(std::string const &fname) { debug_file = std::make_shared<HD5::Writer>(fname); }

auto IsDebugging() -> bool { return debug_file != nullptr; }

void EndDebugging() { debug_file.reset(); }

template <typename Scalar, int N>
void Tensor(std::string const &nameIn, Sz<N> const &shape, Scalar const *data, HD5::DimensionNames<N> const &dimNames)
{
  if (debug_file) {
    Index       count = 0;
    std::string name = nameIn;
    while (debug_file->exists(name)) {
      count++;
      name = fmt::format("{}-{}", nameIn, count);
    }
    debug_file->writeTensor(name, shape, data, dimNames);
  }
}

template void Tensor(std::string const &, Sz<1> const &shape, float const *data, HD5::DimensionNames<1> const &);
template void Tensor(std::string const &, Sz<2> const &shape, float const *data, HD5::DimensionNames<2> const &);
template void Tensor(std::string const &, Sz<3> const &shape, float const *data, HD5::DimensionNames<3> const &);
template void Tensor(std::string const &, Sz<4> const &shape, float const *data, HD5::DimensionNames<4> const &);
template void Tensor(std::string const &, Sz<5> const &shape, float const *data, HD5::DimensionNames<5> const &);
template void Tensor(std::string const &, Sz<3> const &shape, Cx const *data, HD5::DimensionNames<3> const &);
template void Tensor(std::string const &, Sz<4> const &shape, Cx const *data, HD5::DimensionNames<4> const &);
template void Tensor(std::string const &, Sz<5> const &shape, Cx const *data, HD5::DimensionNames<5> const &);
template void Tensor(std::string const &, Sz<6> const &shape, Cx const *data, HD5::DimensionNames<6> const &);

} // namespace Log
} // namespace rl
