#include "signals.hpp"

#include "../log/log.hpp"
#include <cassert>
#include <csignal>

namespace rl {

namespace {
int  interruptLevel = 0;
bool received = false;
} // namespace

void Handler(int sig)
{
  if (sig == SIGINT) {
    if (received) { throw Log::Failure("Signal", "Second SIGINT received, terminating"); }
    Log::Print("Signal", "SIGINT received, will terminate when current iteration finishes. Press Ctrl-C again to terminate now.");
    received = true;
  } else {
    throw Log::Failure("Signal", "Unknown signal {} received, aborting", sig);
  }
}

void PushInterrupt()
{
  if (interruptLevel == 0) { std::signal(SIGINT, Handler); }
  interruptLevel++;
}

void PopInterrupt()
{
  interruptLevel--;
  if (interruptLevel == 0) { std::signal(SIGINT, SIG_DFL); }
}

bool InterruptReceived() { return received; }

} // namespace rl