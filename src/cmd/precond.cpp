#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "precond.hpp"

using namespace rl;

int main_precond(args::Subparser &parser)
{
  args::Positional<std::string> trajFile(parser, "F", "File to read trajectory from");
  args::Positional<std::string> preFile(parser, "F", "File to save pre-conditioner to");
  ParseCommand(parser, trajFile);
  HD5::Reader reader(trajFile.Get());
  HD5::Writer writer(preFile.Get());
  writer.writeTensor(KSpaceSingle(Trajectory(reader)), HD5::Keys::Precond);
  return EXIT_SUCCESS;
}
