#include "basis.hpp"

#include "algo/decomp.hpp"
#include "algo/stats.hpp"
#include "io/reader.hpp"

namespace rl {

auto IdBasis() -> Basis
{
  Basis id(1, 1, 1);
  id.setConstant(1.f);
  return id;
}

auto ReadBasis(std::string const &basisFile) -> Basis
{
  if (basisFile.empty()) {
    return IdBasis();
  } else {
    HD5::Reader basisReader(basisFile);
    Basis       b = basisReader.readTensor<Basis>(HD5::Keys::Basis);
    return b;
  }
}

} // namespace rl
