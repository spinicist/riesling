#include "recon.h"

#include "../sdc.h"
#include "../threads.h"

ReconOp::ReconOp(
  Trajectory const &traj,
  float const os,
  bool const kb,
  bool const fast,
  std::string const sdc,
  Cx4 const &maps,
  Log &log)
  : gridder_{make_grid(traj, os, kb, fast, log)}
  , grid_{gridder_->newMultichannel(maps.dimension(0))}
  , sense_{maps, grid_.dimensions()}
  , apo_{gridder_->apodization(sense_.dimensions())}
  , fft_{grid_, log}
  , log_{log}
{
  if (sense_.channels() != traj.info().channels) {
    Log::Fail(
      "Number of SENSE channels {} did not match header channels {}",
      sense_.channels(),
      traj.info().channels);
  }

  SDC::Choose(sdc, traj, gridder_, log);
}

Sz3 ReconOp::dimensions() const
{
  return sense_.dimensions();
}

Sz3 ReconOp::outputDimensions() const
{
  return gridder_->outputDimensions();
}

void ReconOp::setPreconditioning(float const p)
{
  gridder_->setSDCExponent(p);
}

void ReconOp::calcToeplitz(Info const &info)
{
  log_.info("Calculating Töplitz embedding");
  transfer_ = gridder_->newMultichannel(1);
  Cx3 ones(1, info.read_points, info.spokes_total());
  ones.setConstant({1.0f});
  gridder_->Adj(ones, transfer_);
}

void ReconOp::A(Input const &x, Output &y) const
{
  auto dev = Threads::GlobalDevice();
  auto const &start = log_.now();
  Cx3 apodised(x.dimensions());
  apodised.device(dev) = x / apo_.cast<Cx>();
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
  y.device(dev) = y / apo_.cast<Cx>();
  log_.debug("Decode: {}", log_.toNow(start));
}

void ReconOp::AdjA(Input const &x, Input &y) const
{
  if (transfer_.size() == 0) {
    Log::Fail("Töplitz embedding not calculated");
  }
  auto dev = Threads::GlobalDevice();
  auto const start = log_.now();
  sense_.A(x, grid_);
  fft_.forward(grid_);
  grid_.device(dev) = grid_ * transfer_.broadcast(Sz4{grid_.dimension(0), 1, 1, 1});
  fft_.reverse(grid_);
  sense_.Adj(grid_, y);
  log_.debug("Töplitz embedded: {}", log_.toNow(start));
}
