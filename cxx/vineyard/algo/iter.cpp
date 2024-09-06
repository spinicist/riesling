#include "iter.hpp"

#include "log.hpp"
#include "sys/signals.hpp"
#include <filesystem>

namespace rl::Iterating {

void Starting() { PushInterrupt(); }

void Finished() { PopInterrupt(); }

auto ShouldStop() -> bool
{
  if (InterruptReceived()) {
    return true;
  } else if (std::filesystem::exists(".stop")) {
    Log::Print(".stop file detected, halting iterations");
    return true;
  } else {
    return false;
  }
}

} // namespace rl::Iterating
