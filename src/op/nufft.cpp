#include "nufft.h"

#include "../sdc.h"
#include "../tensorOps.h"
#include "../threads.h"

NufftOp::NufftOp(
  GridBase *gridder, Index const nc, Index const ne, Eigen::Array3l const mat, Log &log)
  : gridder_{gridder}
  , grid_{Sz5{nc,
              ne,
              gridder_->mapping().cartDims[0],
              gridder_->mapping().cartDims[1],
              gridder_->mapping().cartDims[2]}}
  , fft_{grid_, log}
  , pad_{Sz5{nc, ne, mat[0], mat[1], mat[2]}, grid_.dimensions()}
  , log_{log}
{
}

NufftOp::InputDims NufftOp::inputDimensions() const
{
  return pad_.inputDimensions();
}

NufftOp::OutputDims NufftOp::outputDimensions() const
{
  return gridder_->outputDimensions();
}

void NufftOp::A(Input const &x, Output &y) const
{
  auto const &start = log_.now();

  pad_.A(x, grid_); // Pad takes care of apodization
  fft_.forward(grid_);
  gridder_->A(grid_, y);
  log_.debug("Encode: {}", log_.toNow(start));
}

void NufftOp::Adj(Output const &x, Input &y) const
{
  auto const &start = log_.now();
  gridder_->Adj(x, grid_);
  fft_.reverse(grid_);
  pad_.Adj(grid_, y); // This is a crop which takes care of apodization
  log_.debug("Decode: {}", log_.toNow(start));
}

void NufftOp::AdjA(Input const &x, Input &y) const
{
  Output temp(gridder_->mapping().noncartDims);
  A(x, temp);
  Adj(temp, y);
}
