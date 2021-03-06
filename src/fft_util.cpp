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
  fftwf_set_timelimit(60.0);
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

void SetTimelimit(double time)
{
  fftwf_set_timelimit(time);
}

/*
 * Phase factors for FFT shifting
 *
 * I am indebted to Martin Uecker for putting this code in BART
 */
Cx1 Phase(long const sz)
{
  long const c = sz / 2;
  double const shift = (double)c / sz;
  Rd1 ii(sz);
  std::iota(ii.data(), ii.data() + ii.size(), 0.);
  auto const s = ((ii - ii.constant(c / 2.)) * ii.constant(shift));
  Cxd1 const ph = ((s - s.floor()) * s.constant(2. * M_PI)).cast<Cxd>();
  Cx1 const factors = (ph * ph.constant(Cxd{0., 1.})).exp().cast<Cx>();
  return factors;
}

} // namespace FFT