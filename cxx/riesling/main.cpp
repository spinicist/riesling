#include "log.hpp"
#include "parse_args.hpp"

using namespace rl;

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::Group          commands(parser, "COMMANDS");

#define COMMAND(NM, CMD, DESC)                                                                                                 \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(commands, CMD, DESC, &main_##NM);

  COMMAND(basis, "basis", "Create a subspace basis");
  COMMAND(data, "data", "Manipulate riesling files");
#ifdef BUILD_MONTAGE
  COMMAND(montage, "montage", "Make beautiful output images");
#endif
  COMMAND(op, "op", "Linear Operators");
  COMMAND(recon, "recon", "Reconstruction");
  COMMAND(sense, "sense", "Sensitivity maps");
  COMMAND(util, "util", "Utilities");
  COMMAND(version, "version", "Print version number");
  args::GlobalOptions globals(parser, global_group);
  try {
    parser.ParseCLI(argc, argv);
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, "{}\n", parser.Help());
    exit(EXIT_SUCCESS);
  } catch (args::Error &e) {
    fmt::print(stderr, "{}\n", parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", e.what());
    exit(EXIT_FAILURE);
  } catch (Log::Failure &f) {
    Log::End();
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
