#include "recon-basis.h"

#include "../io_hd5.h"
#include "../sdc.h"
#include "../threads.h"
#include "grid.h"

ReconBasisOp::ReconBasisOp(
    Trajectory const &traj,
    float const os,
    bool const kb,
    bool const fast,
    std::string const sdc,
    Cx4 &maps,
    R2 &basis,
    Log &log)
    : gridder_{make_grid_basis(traj, os, kb, fast, basis, log)}
    , grid_{traj.info().channels,
            gridder_->basis().dimension(1),
            gridder_->dimension(0),
            gridder_->dimension(1),
            gridder_->dimension(2)}
    , sense_{maps, grid_.dimensions()}
    , apo_{gridder_->apodization(sense_.dimensions())}
    , fft_{grid_, log}
    , log_{log}
{
  if (sense_.channels() != traj.info().channels) {
    Log::Fail(
        "Number of SENSE channels {} did not match data channels {}",
        sense_.channels(),
        traj.info().channels);
  }

  auto grid1 = make_grid(gridder_->mapping(), kb, fast, log);
  SDC::Choose(sdc, traj, grid1, gridder_, log);
}

Sz3 ReconBasisOp::dimensions() const
{
  return sense_.dimensions();
}

Sz3 ReconBasisOp::outputDimensions() const
{
  return gridder_->outputDimensions();
}

void ReconBasisOp::setPreconditioning(float const p)
{
  gridder_->setSDCExponent(p);
}

void ReconBasisOp::A(Input const &x, Output &y) const
{
  auto dev = Threads::GlobalDevice();
  auto const &start = log_.now();

  using FixOne = Eigen::type2index<1>;
  long const nB = x.dimension(0);
  Eigen::IndexList<FixOne, int, int, int> rshA;
  Eigen::IndexList<int, FixOne, FixOne, FixOne> brdA;
  rshA.set(1, apo_.dimension(0));
  rshA.set(2, apo_.dimension(1));
  rshA.set(3, apo_.dimension(2));
  brdA.set(0, nB);

  Cx4 apodised(x.dimensions());
  apodised.device(dev) = x / apo_.cast<Cx>().reshape(rshA).broadcast(brdA);
  sense_.A(apodised, grid_);
  fft_.forward(grid_);
  gridder_->A(grid_, y);
  log_.debug("Encode: {}", log_.toNow(start));
}

void ReconBasisOp::Adj(Output const &x, Input &y) const
{
  auto dev = Threads::GlobalDevice();
  auto const &start = log_.now();
  gridder_->Adj(x, grid_);
  fft_.reverse(grid_);
  sense_.Adj(grid_, y);

  using FixOne = Eigen::type2index<1>;
  long const nB = y.dimension(0);
  Eigen::IndexList<FixOne, int, int, int> rshA;
  Eigen::IndexList<int, FixOne, FixOne, FixOne> brdA;
  rshA.set(1, apo_.dimension(0));
  rshA.set(2, apo_.dimension(1));
  rshA.set(3, apo_.dimension(2));
  brdA.set(0, nB);
  y.device(dev) = y / apo_.cast<Cx>().reshape(rshA).broadcast(brdA);
  log_.debug("Decode: {}", log_.toNow(start));
}
