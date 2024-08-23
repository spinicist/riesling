#include "types.hpp"

#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "precon.hpp"

using namespace rl;

void main_precon(args::Subparser &parser)
{
  args::Positional<std::string> trajFile(parser, "INPUT", "File to read trajectory from");
  args::Positional<std::string> preFile(parser, "OUTPUT", "File to save pre-conditioner to");
  args::ValueFlag<float>        preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"bias"}, 1.f);
  args::Flag                    vcc(parser, "VCC", "Include VCC", {"vcc"});
  args::ValueFlag<std::string>  basisFile(parser, "BASIS", "File to read basis from", {"basis", 'b'});
  ParseCommand(parser, trajFile);
  HD5::Reader reader(trajFile.Get());
  HD5::Writer writer(preFile.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto const basis = LoadBasis(basisFile.Get());
  auto        M = KSpaceSingle(traj, basis.get(), vcc, preBias.Get());
  writer.writeTensor(HD5::Keys::Weights, M.dimensions(), M.data(), {"sample", "trace"});
  Log::Print("Finished {}", parser.GetCommand().Name());
}
