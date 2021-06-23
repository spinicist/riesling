#include "apodizer.h"

#include "cropper.h"
#include "fft_plan.h"

Apodizer::Apodizer(Kernel *const k, Sz3 const &grid, Sz3 const &img, Log &log)
    : log_(log)
{
  Cx3 temp(grid);
  FFT::ThreeD fft(temp, log_);
  temp.setZero();
  Crop3(temp, k->size()) = k->kspace(Point3::Zero()).cast<Cx>();
  fft.reverse(temp);
  y_ = Crop3(R3(temp.real()), img);
  float const scale = sqrt(std::accumulate(grid.cbegin(), grid.cend(), 1, std::multiplies<long>()));
  log.info(
      FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(y_.dimensions(), ","), scale);
  y_.device(Threads::GlobalDevice()) = y_ * y_.constant(scale);
  log.image(y_, "apodization.nii");
}

void Apodizer::apodize(Cx3 &x)
{
  log_.info("Apodizing...");
  x.device(Threads::GlobalDevice()) = x * y_.cast<Cx>();
}

void Apodizer::deapodize(Cx3 &x)
{
  log_.info("De-apodizing...");
  x.device(Threads::GlobalDevice()) = x / y_.cast<Cx>();
}
