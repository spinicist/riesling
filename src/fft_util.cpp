#include "fft_util.h"

#include "tensorOps.h"
#include "threads.h"
#include <filesystem>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

namespace FFT {

std::filesystem::path WisdomPath()
{
  struct passwd *pw = getpwuid(getuid());
  const char *homedir = pw->pw_dir;
  return std::filesystem::path(homedir) / ".riesling-wisdom";
}

void Start(Log &log)
{
  fftwf_init_threads();
  fftwf_make_planner_thread_safe();
  if (fftwf_import_wisdom_from_filename(WisdomPath().string().c_str())) {
    log.info("Read wisdom successfully");
  } else {
    log.info("Could not read wisdom");
  }
}

void End(Log &log)
{
  auto const &wp = WisdomPath();
  if (fftwf_export_wisdom_to_filename(wp.string().c_str())) {
    log.info(FMT_STRING("Saved wisdom to {}"), wp.string());
  } else {
    log.info("Failed to save wisdom");
  }
  // Get use after free errors if this is called before fftw_plan_destroy in the
  // destructors
  // fftwf_cleanup_threads();
}

/*
 * Phase factors for FFT shifting
 *
 * I am indebted to Martin Uecker for putting this code in BART
 */
Cxd1 Phase(long const sz)
{
  long const c = sz / 2;
  double const shift = (double)c / sz;
  Rd1 ii(sz);
  std::iota(ii.data(), ii.data() + ii.size(), 0.);
  auto const ph = ((ii - ii.constant(c / 2.)) * ii.constant(shift));
  auto const factors =
      ph.unaryExpr([](double const x) { return std::polar(1., (x - std::floor(x)) * 2. * M_PI); });
  return factors;
}

} // namespace FFT