#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "precon.hpp"

using namespace rl;

void main_precon(args::Subparser &parser)
{
  args::Positional<std::string> trajFile(parser, "INPUT", "File to read trajectory from");
  args::Positional<std::string> preFile(parser, "OUTPUT", "File to save pre-conditioner to");
  args::ValueFlag<float>        preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"bias", 'b'}, 1.f);
  args::ValueFlag<std::string>  basisFile(parser, "BASIS", "File to read basis from", {"basis"});
  ParseCommand(parser, trajFile);
  HD5::Reader reader(trajFile.Get());
  HD5::Writer writer(preFile.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        M = KSpaceSingle(traj, ReadBasis(basisFile.Get()), preBias.Get());
  writer.writeTensor(HD5::Keys::Weights, M.dimensions(), M.data());
  Log::Print("Finished {}", parser.GetCommand().Name());
}
