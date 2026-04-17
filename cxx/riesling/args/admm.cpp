#include "admm.hpp"

using namespace rl;

ADMMArgs::ADMMArgs(args::Subparser &parser)
  : in_its0(parser, "ITS", "Initial inner iterations (64)", {"max-its0"}, 64)
  , in_its1(parser, "ITS", "Subsequent inner iterations (64)", {"max-its1"}, 64)
  , atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f)
  , btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f)
  , ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f)
  , ε(parser, "ε", "ADMM convergence tolerance (1e-3)", {"eps"}, 1.e-3f)
  , outerIts(parser, "ITS", "ADMM max iterations (64)", {"max-outer-its"}, 64)
  , restart(parser, "R", "Restart average iterate (8)", {"restart"}, 8)
  , ρ(parser, "ρ", "ADMM starting penalty parameter ρ (default 1)", {"rho"}, 1.f)
  , updateρ(parser, "U", "Update ρ when restarting", {"rho-update"})
  , τ(parser, "τ", "Fallback ρ update", {"tau"}, 10.f)
{
}

auto ADMMArgs::Get() -> rl::ADMM::Opts
{
  return rl::ADMM::Opts{.iters0 = in_its0.Get(),
                        .iters1 = in_its1.Get(),
                        .aTol = atol.Get(),
                        .bTol = btol.Get(),
                        .cTol = ctol.Get(),
                        .ε = ε.Get(),
                        .outerLimit = outerIts.Get(),
                        .restart = restart.Get(),
                        .ρ = ρ.Get(),
                        .updateρ = updateρ.Get(),
                        .τ = τ.Get()};
}
