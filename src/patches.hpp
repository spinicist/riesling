#include "types.hpp"

namespace rl {

using PatchFunction = std::function<Cx4(Cx4 const &)>;

void Patches(
  Index const patchSize,
  Index const windowSize,
  bool const shift,
  PatchFunction const &apply,
  Eigen::TensorMap<Cx4 const> const &x,
  Eigen::TensorMap<Cx4> &y);

} // namespace rl
