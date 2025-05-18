#include "args.hpp"
#include "rl/log/log.hpp"
using namespace rl;

namespace {
std::unordered_map<int, Log::Display> levelMap{{0, Log::Display::None},
                                               {1, Log::Display::Ephemeral},
                                               {2, Log::Display::Low},
                                               {3, Log::Display::Mid},
                                               {4, Log::Display::High}};

} // namespace

#define COMMAND(PARSER, NM, CMD, DESC)                                                                                         \
  void          main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(PARSER, CMD, DESC, &main_##NM);

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("GEWURZ");

  args::HelpFlag                   help(parser, "H", "Show this help message", {'h', "help"});
  args::MapFlag<int, Log::Display> verbosity(parser, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Display::Low);

  COMMAND(parser, dft, "dft", "Apply DFT");
  try {
    parser.ParseCLI(argc, argv);
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, "{}\n", parser.Help());
    return EXIT_SUCCESS;
  } catch (args::Error &e) {
    fmt::print(stderr, "{}\n", parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", e.what());
    return EXIT_FAILURE;
  } catch (Log::Failure &f) {
    Log::Fail(f);
    Log::End();
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    Log::Fail(Log::Failure("None", "{}", e.what()));
    Log::End();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
