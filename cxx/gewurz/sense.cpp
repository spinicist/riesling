#include "sense.hpp"

#include "rl/sense/sense.hpp"

namespace gw {

auto GetSENSE(std::string const &path, rl::HD5::Shape<3> const mat) -> std::vector<std::complex<float>>
{

  rl::HD5::Reader reader(path);
  rl::Cx5 const   kernels = reader.readTensor<rl::Cx5>();
  rl::Sz3 mmat;
  std::copy_n(mat.begin(), 3, mmat.begin());
  rl::Cx5 const   maps = rl::SENSE::KernelsToMaps(kernels, mmat, 1.f);

  std::vector<std::complex<float>> s(maps.size());
  std::copy_n(maps.data(), maps.size(), s.begin());

  return s;
}

} // namespace gw