#include "recon-basis.h"

#include "../io_hd5.h"
#include "../sdc.h"
#include "../threads.h"
#include "grid.h"

ReconBasisOp::ReconBasisOp(GridBase *gridder, Cx4 const &maps, Log &log)
  : gridder_{gridder}
  , grid_{gridder_->inputDimensions(maps.dimension(0))}
  , sense_{maps, gridder_.cartDimensions()}
  , apo_{gridder_->apodization(sense_.inputDimensions())}
  , fft_{grid_, log}
  , log_{log}
{
}

Sz3 ReconBasisOp::dimensions() const
{
  return sense_.dimensions();
}

Sz3 ReconBasisOp::outputDimensions() const
{
  return gridder_->outputDimensions();
}

void ReconBasisOp::calcToeplitz(Info const &info)
{
  log_.info("Calculating Töplitz embedding");
  transfer_.resize(gridder_->inputDimensions(1));
  transfer_.setConstant(1.f);
  Cx3 tf(1, info.read_points, info.spokes_total());
  gridder_->A(transfer_, tf);
  gridder_->Adj(tf, transfer_);
}

void ReconBasisOp::A(Input const &x, Output &y) const
{
  auto dev = Threads::GlobalDevice();
  auto const &start = log_.now();

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

void ReconBasisOp::AdjA(Input const &x, Input &y) const
{
  if (transfer_.size() == 0) {
    Log::Fail("Töplitz embedding not calculated");
  }
  auto dev = Threads::GlobalDevice();
  auto const start = log_.now();
  sense_.A(x, grid_);
  fft_.forward(grid_);
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brd;
  brd.set(0, grid_.dimension(0));
  grid_.device(dev) = grid_ * transfer_.broadcast(brd);
  fft_.reverse(grid_);
  sense_.Adj(grid_, y);
  log_.debug("Töplitz embedded: {}", log_.toNow(start));
}
