#include "types.hpp"

#include "algo/stats.hpp"
#include "fft.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/fft.hpp"
#include "op/pad.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_sense_maps(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file"),
    tname(parser, "FILE", "Target HD5 file for reconstruction"), oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<float> osamp(parser, "O", "Grid oversampling factor (2)", {"osamp"}, 2.f);
  Array3fFlag            fov(parser, "SENSE-FOV", "SENSE FOV (default header FOV)", {"sense-fov"}, Eigen::Array3f::Zero());

  ParseCommand(parser, iname, oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(tname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);

  HD5::Reader kreader(iname.Get());
  Cx5 const   kernels = kreader.readTensor<Cx5>();
  Cx5 const   maps = SENSE::KernelsToMaps(kernels, traj.matrixForFOV(fov.Get()), osamp.Get());
  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Data, maps.dimensions(), maps.data(), HD5::Dims::SENSE);
  Log::Print(cmd, "Finished");
}
