#include "types.hpp"

namespace rl {

using PatchFunction = std::function<Cx5(Cx5 const &)>;

void Patches(
  Index const patchSize, Index const windowSize, bool const shift, PatchFunction const &apply, Cx5CMap x, Cx5Map y);

} // namespace rl
