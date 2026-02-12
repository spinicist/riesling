#include "types.hpp"

namespace rl {

template <int D> using PatchFunction = std::function<void(CxN<3 + D> const &, CxN<3 + D> &)>;

template <int D> void Patches(Index const             patchSize,
                              Index const             windowSize,
                              bool const              shift,
                              PatchFunction<D> const &apply,
                              CxNCMap<3 + D>          x,
                              CxNMap<3 + D>           y);

} // namespace rl
