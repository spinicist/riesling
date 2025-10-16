#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/precon.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_precon(args::Subparser &parser)
{
  GridArgs<3>                   gridArgs(parser);
  args::Positional<std::string> trajFile(parser, "INPUT", "File to read trajectory from");
  args::Positional<std::string> preFile(parser, "OUTPUT", "File to save pre-conditioner to");
  args::ValueFlag<float>        preλ(parser, "λ", "Preconditioner regularization (0)", {"precon-l"}, 0.f);
  args::ValueFlag<std::string>  sfile(parser, "S", "Load SENSE kernels from file", {"sense"});
  args::ValueFlag<std::string>  basisFile(parser, "B", "Read basis from file", {"basis", 'b'});
  ParseCommand(parser, trajFile);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(trajFile.Get());
  HD5::Writer writer(preFile.Get());
  Trajectory  traj(reader, reader.readStruct<Info>(HD5::Keys::Info).voxel_size);
  auto const  basis = LoadBasis(basisFile.Get());
  if (sfile) {
    HD5::Reader senseReader(sfile.Get());
    Cx5 const   skern = senseReader.readTensor<Cx5>(HD5::Keys::Data);
    Cx5 const   smaps = SENSE::KernelsToMaps(skern, traj.matrixForFOV(gridArgs.fov.Get()), gridArgs.osamp.Get());
    auto const  M = KSpaceMulti(smaps, gridArgs.Get(), traj, preλ.Get(), basis.get());
    writer.writeTensor(HD5::Keys::Weights, M.dimensions(), M.data(), {"channel", "sample", "trace"});
  } else {
    auto const M = KSpaceSingle(gridArgs.Get(), traj, preλ.Get(), basis.get());
    writer.writeTensor(HD5::Keys::Weights, M.dimensions(), M.data(), {"sample", "trace"});
  }

  Log::Print(cmd, "Finished");
}
