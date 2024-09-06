#pragma once
/*
 * Infrastructure for dealing with whether to continue iterating in optimizers
 */
namespace rl::Iterating {
void Starting();
void Finished();
auto ShouldStop() -> bool;
} // namespace rl::Iterating
