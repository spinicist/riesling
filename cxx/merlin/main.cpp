#include "rl/log.hpp"
#include "args.hpp"

extern args::Group    global_group;
extern args::HelpFlag help;
extern args::Flag     verbose;

using namespace rl;

#define COMMAND(PARSER, NM, CMD, DESC)                                                                                         \
  void          main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(PARSER, CMD, DESC, &main_##NM);

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("MERLIN");

  args::Group grp(parser, "MERLIN");
  COMMAND(grp, move, "move", "Shift and rotate non-cartesian data in FOV");
  COMMAND(grp, navs, "basis-navs", "Create a basis for navigators");

  args::GlobalOptions globals(parser, global_group);
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
