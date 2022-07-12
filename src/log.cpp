#include "log.h"

#include "fmt/chrono.h"
#include "indicators/progress_bar.hpp"
#include "io/hd5.hpp"
#include "tensorOps.h"

namespace rl {
namespace Log {

Failure::Failure(std::string const &msg)
  : std::runtime_error(msg)
{
}

namespace {
Level log_level = Level::None;
std::unique_ptr<HD5::Writer> debug_file = nullptr;
std::unique_ptr<indicators::ProgressBar> progress = nullptr;
Index progressTarget;
std::atomic<Index> progressAmount;
} // namespace

Level CurrentLevel()
{
  return log_level;
}

void SetLevel(Level const l)
{
  log_level = l;
}

void SetDebugFile(std::string const &fname)
{
  debug_file = std::make_unique<HD5::Writer>(fname);
}

void End()
{
  debug_file.reset();
  log_level = Level::None;
}

void lprint(fmt::string_view fstr, fmt::format_args args)
{
  if (log_level >= Level::Info) {
    auto const t = std::chrono::system_clock::now();
    fmt::print(stderr, FMT_STRING("[{:%H:%M:%S}] {}\n"), fmt::localtime(t), fmt::vformat(fstr, args));
  }
}

void ldebug(fmt::string_view fstr, fmt::format_args args)
{
  if (log_level >= Level::Debug) {
    auto const t = std::chrono::system_clock::now();
    fmt::print(stderr, FMT_STRING("[{:%H:%M:%S}] {}\n"), fmt::localtime(t), fmt::vformat(fstr, args));
  }
}

void lfail(fmt::string_view fstr, fmt::format_args args)
{
  auto const t = std::chrono::system_clock::now();
  auto const msg = fmt::format(FMT_STRING("[{:%H:%M:%S}] {}\n"), fmt::localtime(t), fmt::vformat(fstr, args));
  if (log_level > Level::Testing) {
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), msg);
  }
  throw Failure(msg);
}

void StartProgress(Index const amount, std::string const &text)
{
  if (log_level >= Level::Progress) {
    progress = std::make_unique<indicators::ProgressBar>(
      indicators::option::BarWidth{80},
      indicators::option::PrefixText{text},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true});
    progressTarget = amount - 1;
    progressAmount = 0;
  }
}

void StopProgress()
{
  if (progress) {
    if (!progress->is_completed()) {
      progress->mark_as_completed();
    }
    progress = nullptr;
  }
}

void Tick()
{
  if (progress) {
    progressAmount++;
    float const percent = (100.f * progressAmount) / progressTarget;
    if ((percent - progress->current() > 1.f) && (percent <= 100.f)) {
      progress->set_progress(percent);
    }
  }
}

Time Now()
{
  return std::chrono::high_resolution_clock::now();
}

std::string ToNow(Log::Time const t1)
{
  using ms = std::chrono::milliseconds;
  auto const t2 = std::chrono::high_resolution_clock::now();
  auto const diff = std::chrono::duration_cast<ms>(t2 - t1).count();
  return fmt::format(FMT_STRING("{} ms"), diff);
}

template <typename Scalar, int ND>
void Tensor(Eigen::Tensor<Scalar, ND> const &img, std::string const &nameIn)
{
  if (debug_file) {
    Index count = 0;
    std::string name = nameIn;
    while (debug_file->exists(name)) {
      count++;
      name = fmt::format("{}-{}", nameIn, count);
    }
    debug_file->writeTensor(img, name);
  }
}

template void Tensor(R3 const &, std::string const &);
template void Tensor(Cx3 const &, std::string const &);
template void Tensor(Cx4 const &, std::string const &);
template void Tensor(Cx5 const &, std::string const &);
template void Tensor(Cx6 const &, std::string const &);

} // namespace Log
} // namespace rl
