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

template <typename Scalar, size_t N>
void Tensor(std::string const &nameIn, HD5::Shape<N> const &shape, Scalar const *data, HD5::DNames<N> const &dimNames)
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

template void Tensor(std::string const &, HD5::Shape<1> const &shape, float const *data, HD5::DNames<1> const &);
template void Tensor(std::string const &, HD5::Shape<2> const &shape, float const *data, HD5::DNames<2> const &);
template void Tensor(std::string const &, HD5::Shape<3> const &shape, float const *data, HD5::DNames<3> const &);
template void Tensor(std::string const &, HD5::Shape<4> const &shape, float const *data, HD5::DNames<4> const &);
template void Tensor(std::string const &, HD5::Shape<5> const &shape, float const *data, HD5::DNames<5> const &);
template void Tensor(std::string const &, HD5::Shape<3> const &shape, Cx const *data, HD5::DNames<3> const &);
template void Tensor(std::string const &, HD5::Shape<4> const &shape, Cx const *data, HD5::DNames<4> const &);
template void Tensor(std::string const &, HD5::Shape<5> const &shape, Cx const *data, HD5::DNames<5> const &);
template void Tensor(std::string const &, HD5::Shape<6> const &shape, Cx const *data, HD5::DNames<6> const &);

} // namespace Log
} // namespace rl
