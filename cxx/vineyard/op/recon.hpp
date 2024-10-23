#pragma once

#include "io/reader.hpp"
#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "sense/sense.hpp"
#include "sys/args.hpp"

/*
 *  Full recon operators
 */

namespace rl {
namespace Recon {

auto Choose(GridOpts<3> const &gridOpts, SENSE::Opts &senseOpts, Trajectory const &traj, Basis::CPtr basis, Cx5 const &data)
  -> TOps::TOp<Cx, 5, 5>::Ptr;

auto Single(GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr basis)
  -> TOps::TOp<Cx, 5, 5>::Ptr;

auto SENSE(GridOpts<3> const &gridOpts,
           Trajectory const  &traj,
           Index const        nSlab,
           Index const        nTime,
           Basis::CPtr        basis,
           Cx5 const         &smaps) -> TOps::TOp<Cx, 5, 5>::Ptr;

auto Decant(GridOpts<3> const &gridOpts,
            Trajectory const  &traj,
            Index const        nSlab,
            Index const        nTime,
            Basis::CPtr        basis,
            Cx5 const         &kernels,
            Sz3 const         &matrix) -> TOps::TOp<Cx, 5, 5>::Ptr;

} // namespace Recon
} // namespace rl