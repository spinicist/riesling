#include "recon.h"

#include "../sdc.h"
#include "../tensorOps.h"
#include "../threads.h"

ReconOp::ReconOp(GridBase *gridder, Cx4 const &maps, Log &log)
  : gridder_{gridder}
  , grid_{gridder_->inputDimensions(maps.dimension(0))}
  , sense_{maps, grid_.dimensions()}
  , apo_{gridder_->apodization(Sz3{maps.dimension(1), maps.dimension(2), maps.dimension(3)})}
  , fft_{grid_, log}
  , log_{log}
{
}

ReconOp::InputDims ReconOp::inputDimensions() const
{
  return sense_.inputDimensions();
}

ReconOp::OutputDims ReconOp::outputDimensions() const
{
  return gridder_->outputDimensions();
}

void ReconOp::calcToeplitz(Info const &info)
{
  log_.info("Calculating Töplitz embedding");
  transfer_.resize(gridder_->inputDimensions(1));
  transfer_.setConstant(1.f);
  Cx3 tf(1, info.read_points, info.spokes_total());
  gridder_->A(transfer_, tf);
  gridder_->Adj(tf, transfer_);
}

void ReconOp::A(Input const &x, Output &y) const
{
  auto dev = Threads::GlobalDevice();
  auto const &start = log_.now();

  Index const nB = x.dimension(0);
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

void ReconOp::Adj(Output const &x, Input &y) const
{
  auto dev = Threads::GlobalDevice();
  auto const &start = log_.now();
  gridder_->Adj(x, grid_);
  fft_.reverse(grid_);
  sense_.Adj(grid_, y);

  Index const nB = y.dimension(0);
  Eigen::IndexList<FixOne, int, int, int> rshA;
  Eigen::IndexList<int, FixOne, FixOne, FixOne> brdA;
  rshA.set(1, apo_.dimension(0));
  rshA.set(2, apo_.dimension(1));
  rshA.set(3, apo_.dimension(2));
  brdA.set(0, nB);
  y.device(dev) = y / apo_.cast<Cx>().reshape(rshA).broadcast(brdA);
  log_.debug("Decode: {}", log_.toNow(start));
}

void ReconOp::AdjA(Input const &x, Input &y) const
{
  if (transfer_.size() == 0) {
    Output temp;
    A(x, temp);
    Adj(temp, y);
  } else {
    auto dev = Threads::GlobalDevice();
    auto const start = log_.now();
    sense_.A(x, grid_);
    fft_.forward(grid_);
    Eigen::IndexList<int, FixOne, FixOne, FixOne> brd;
    brd.set(0, grid_.dimension(0));
    grid_.device(dev) = grid_ * transfer_.broadcast(brd);
    fft_.reverse(grid_);
    sense_.Adj(grid_, y);
    log_.debug("Töplitz embedded: {}", log_.toNow(start));
  }
}
