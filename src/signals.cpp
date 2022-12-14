#include "signals.hpp"

#include "log.hpp"
#include <cassert>
#include <csignal>

namespace rl {

namespace {
int interruptLevel = 0;
bool received = false;
} // namespace

void Handler(int sig)
{
  if (received) {
    std::abort();
  } else {
    Log::Print("SIGINT received, will terminate when current iteration finishes. Press Ctrl-C again to terminate now.");
    received = true;
  }
}

void PushInterrupt()
{
  if (interruptLevel == 0) {
    std::signal(SIGINT, Handler);
  }
  interruptLevel++;
}

void PopInterrupt()
{
  interruptLevel--;
  if (interruptLevel == 0) {
    std::signal(SIGINT, SIG_DFL);
  }
}

bool InterruptReceived() { return received; }

} // namespace rl