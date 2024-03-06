#include "cropper.h"

#include <ranges>

namespace rl {

Cropper::Cropper(Sz3 const matrix, Sz3 const fullSz, Eigen::Array3f const voxelSz, Eigen::Array3f const extent)
{
  for (Index ii = 0; ii < 3; ii++) {
    if (extent[ii] > 0.f) {
      sz_[ii] = std::min(((Index)((extent[ii] / voxelSz[ii]) / 2.f) * 2), fullSz[ii]);
    } else {
      sz_[ii] = std::min(matrix[ii], fullSz[ii]);
    }
  }
  calcStart(fullSz);
  Log::Debug("Cropper start {} size {} full {} extent {} voxel-size {}", st_, sz_, fullSz, extent.transpose(),
             voxelSz.transpose());
}

Cropper::Cropper(Sz3 const &fullSz, Eigen::Array3l const &cropSz)
{
  sz_[0] = cropSz[0];
  sz_[1] = cropSz[1];
  sz_[2] = cropSz[2];
  calcStart(fullSz);
  Log::Debug("Cropper start {} size {}", st_, sz_);
}

Cropper::Cropper(Sz3 const &fullSz, Sz3 const &cropSz)
{
  sz_ = cropSz;
  calcStart(fullSz);
  Log::Debug("Cropper start {} size {}", st_, sz_);
}

void Cropper::calcStart(Sz3 const &fullSz)
{
  // After truncation the -1 makes even and odd sizes line up the way we want
  st_ = Sz3{(fullSz[0] - (sz_[0] - 1)) / 2, (fullSz[1] - (sz_[1] - 1)) / 2, (fullSz[2] - (sz_[2] - 1)) / 2};
  if (std::ranges::any_of(st_, [](Index const x) { return x < 0; })) {
    Log::Fail("Requested crop size {} was larger than available size {}", sz_, fullSz);
  }
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
