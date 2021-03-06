#include "cropper.h"

Cropper::Cropper(Info const &info, Sz3 const &fullSz, float const extent, Log &log)
{
  if (extent < 0.f) {
    std::copy_n(info.matrix.begin(), 3, sz_.begin());
  } else {
    Eigen::Array3l full;
    std::copy_n(&fullSz[0], 3, full.begin());
    // Ensure even, no bigger than the grid
    Eigen::Array3l crop = (((extent / info.voxel_size) / 2).floor() * 2).cast<long>().min(full);
    std::copy_n(crop.begin(), 3, sz_.begin());
  }
  calcStart(fullSz);
  if (info.type == Info::Type::ThreeDStack) {
    st_[2] = 0;
    sz_[2] = info.matrix[2];
  }
  log.debug(FMT_STRING("Cropper start {} size {}"), st_, sz_);
}

Cropper::Cropper(Sz3 const &fullSz, Eigen::Array3l const &cropSz, Log &log)
{
  sz_[0] = cropSz[0];
  sz_[1] = cropSz[1];
  sz_[2] = cropSz[2];
  calcStart(fullSz);
  log.debug(FMT_STRING("Cropper start {} size {}"), st_, sz_);
}

Cropper::Cropper(Sz3 const &fullSz, Sz3 const &cropSz, Log &log)
{
  sz_ = cropSz;
  calcStart(fullSz);
  log.debug(FMT_STRING("Cropper start {} size {}"), st_, sz_);
}

void Cropper::calcStart(Sz3 const &fullSz)
{
  // After truncation the -1 makes even and odd sizes line up the way we want
  st_ = Sz3{(fullSz[0] - (sz_[0] - 1)) / 2,
            (fullSz[1] - (sz_[1] - 1)) / 2,
            (fullSz[2] - (sz_[2] - 1)) / 2};
}

Sz3 Cropper::size() const
{
  return sz_;
}

Sz4 Cropper::dims(long const nChan) const
{
  return Sz4{nChan, sz_[0], sz_[1], sz_[2]};
}

Sz3 Cropper::start() const
{
  return st_;
}

Cx3 Cropper::newImage() const
{
  return Cx3(sz_);
}

Cx4 Cropper::newMultichannel(long const chans) const
{
  return Cx4(chans, sz_[0], sz_[1], sz_[2]);
}

Cx4 Cropper::newSeries(long const vols) const
{
  return Cx4(sz_[0], sz_[1], sz_[2], vols);
}

R3 Cropper::newRealImage() const
{
  return R3(sz_[0], sz_[1], sz_[2]);
}

R4 Cropper::newRealSeries(long const vols) const
{
  return R4(sz_[0], sz_[1], sz_[2], vols);
}