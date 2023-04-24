#include "cropper.h"

namespace rl {

Cropper::Cropper(Sz3 const matrix, Sz3 const fullSz, Eigen::Array3f const voxelSz, float const extent)
{
  if (extent < 0.f) {
    std::transform(
      matrix.begin(), matrix.end(), fullSz.begin(), sz_.begin(), [](Index const a, Index const b) { return std::min(a, b); });
    Log::Print<Log::Level::High>("Cropper start {} size {}", st_, sz_);
  } else {
    Eigen::Array3l full;
    std::copy_n(&fullSz[0], 3, full.begin());
    // Ensure even, no bigger than the grid
    Eigen::Array3l crop = (((extent / voxelSz) / 2).floor() * 2).cast<Index>().min(full);
    std::copy_n(crop.begin(), 3, sz_.begin());
    Log::Print<Log::Level::High>(
      "Cropper start {} size {} full {} extent {} voxel-size {}", st_, sz_, fullSz, extent, voxelSz.transpose());
  }
  calcStart(fullSz);
}

Cropper::Cropper(Sz3 const &fullSz, Eigen::Array3l const &cropSz)
{
  sz_[0] = cropSz[0];
  sz_[1] = cropSz[1];
  sz_[2] = cropSz[2];
  calcStart(fullSz);
  Log::Print<Log::Level::High>("Cropper start {} size {}", st_, sz_);
}

Cropper::Cropper(Sz3 const &fullSz, Sz3 const &cropSz)
{
  sz_ = cropSz;
  calcStart(fullSz);
  Log::Print<Log::Level::High>("Cropper start {} size {}", st_, sz_);
}

void Cropper::calcStart(Sz3 const &fullSz)
{
  // After truncation the -1 makes even and odd sizes line up the way we want
  st_ = Sz3{(fullSz[0] - (sz_[0] - 1)) / 2, (fullSz[1] - (sz_[1] - 1)) / 2, (fullSz[2] - (sz_[2] - 1)) / 2};
}

Sz3 Cropper::size() const { return sz_; }

Sz4 Cropper::dims(Index const nChan) const { return Sz4{nChan, sz_[0], sz_[1], sz_[2]}; }

Sz3 Cropper::start() const { return st_; }

Cx3 Cropper::newImage() const { return Cx3(sz_); }

Cx4 Cropper::newMultichannel(Index const chans) const { return Cx4(chans, sz_[0], sz_[1], sz_[2]); }

Cx4 Cropper::newSeries(Index const vols) const { return Cx4(sz_[0], sz_[1], sz_[2], vols); }

Re3 Cropper::newRealImage() const { return Re3(sz_[0], sz_[1], sz_[2]); }

Re4 Cropper::newRealSeries(Index const vols) const { return Re4(sz_[0], sz_[1], sz_[2], vols); }

} // namespace rl
