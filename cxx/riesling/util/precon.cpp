#include "types.hpp"

#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "precon.hpp"

using namespace rl;

void main_precon(args::Subparser &parser)
{
  GridArgs<3>                   gridArgs(parser);
  args::Positional<std::string> trajFile(parser, "INPUT", "File to read trajectory from");
  args::Positional<std::string> preFile(parser, "OUTPUT", "File to save pre-conditioner to");
  args::ValueFlag<float>        preλ(parser, "BIAS", "Pre-conditioner regularization (1)", {"lambda"}, 1.f);
  args::ValueFlag<std::string>  basisFile(parser, "BASIS", "File to read basis from", {"basis", 'b'});
  args::ValueFlag<std::string>  sfile(parser, "S", "Load SENSE kernels from file", {"sense"});
  ParseCommand(parser, trajFile);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(trajFile.Get());
  HD5::Writer writer(preFile.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto const  basis = LoadBasis(basisFile.Get());
  if (sfile) {
    HD5::Reader senseReader(sfile.Get());
    Cx5 const   skern = senseReader.readTensor<Cx5>(HD5::Keys::Data);
    Cx5 const   smaps = SENSE::KernelsToMaps(skern, traj.matrixForFOV(gridArgs.fov.Get()), gridArgs.osamp.Get());
    auto        M = KSpaceMulti(smaps, gridArgs.Get(), traj, basis.get(), preλ.Get(), 1, 1);
    writer.writeTensor(HD5::Keys::Weights, M->weights().dimensions(), M->weights().data(), {"channel", "sample", "trace"});
  } else {
    auto M = KSpaceSingle(gridArgs.Get(), traj, basis.get(), preλ.Get(), 1, 1, 1);
    writer.writeTensor(HD5::Keys::Weights, M->weights().dimensions(), M->weights().data(), {"sample", "trace"});
  }

  Log::Print(cmd, "Finished");
}
