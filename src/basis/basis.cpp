#include "basis.hpp"

#include "algo/decomp.hpp"
#include "algo/stats.hpp"
#include "io/reader.hpp"

namespace rl {

template <typename Scalar>
auto IdBasis() -> Eigen::Tensor<Scalar, 2>
{
  Eigen::Tensor<Scalar, 2> id(1, 1);
  id.setConstant(1.f);
  return id;
}

template auto IdBasis<float>() -> Basis<float>;
template auto IdBasis<Cx>() -> Basis<Cx>;

auto ReadBasis(std::string const &basisFile) -> Basis<Cx>
{
  if (basisFile.empty()) {
    return IdBasis();
  } else {
    HD5::Reader basisReader(basisFile);
    return basisReader.readTensor<Basis<Cx>>(HD5::Keys::Basis);
  }
}

} // namespace rl
