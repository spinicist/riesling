#include "grid-internal.hpp"

namespace rl {

template std::unique_ptr<GridBase<Cx>>
make_grid_internal<7, 7, Cx>(Kernel const *k, Mapping const &m, Index const nC);

template std::unique_ptr<GridBase<Cx>>
make_grid_internal<7, 7, Cx>(Kernel const *k, Mapping const &m, Index const nC, R2 const &basis);

template std::unique_ptr<GridBase<float>>
make_grid_internal<7, 7, float>(Kernel const *k, Mapping const &m, Index const nC);

template std::unique_ptr<GridBase<float>>
make_grid_internal<7, 7, float>(Kernel const *k, Mapping const &m, Index const nC, R2 const &basis);

} // namespace rl
