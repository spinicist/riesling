#include "grid-internal.hpp"

namespace rl {

template std::unique_ptr<GridBase<Cx>>
rl::make_grid_internal<3, 1, Cx>(Kernel const *k, Mapping const &m, Index const nC);

template std::unique_ptr<GridBase<Cx>>
make_grid_internal<3, 1, Cx>(Kernel const *k, Mapping const &m, Index const nC, R2 const &basis);

template std::unique_ptr<GridBase<float>>
rl::make_grid_internal<3, 1, float>(Kernel const *k, Mapping const &m, Index const nC);

template std::unique_ptr<GridBase<float>>
make_grid_internal<3, 1, float>(Kernel const *k, Mapping const &m, Index const nC, R2 const &basis);

} // namespace rl
