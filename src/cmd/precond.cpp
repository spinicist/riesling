#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "precond.hpp"

using namespace rl;

int main_precond(args::Subparser &parser)
{
  args::Positional<std::string> trajFile(parser, "INPUT", "File to read trajectory from");
  args::Positional<std::string> preFile(parser, "OUTPUT", "File to save pre-conditioner to");
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "File to read basis from", {"basis"});
  ParseCommand(parser, trajFile);
  HD5::Reader reader(trajFile.Get());
  HD5::Writer writer(preFile.Get());
  auto M = KSpaceSingle(Trajectory(reader), ReadBasis(basisFile.Get()));
  writer.writeTensor(HD5::Keys::Precond, M.dimensions(), M.data());
  return EXIT_SUCCESS;
}
