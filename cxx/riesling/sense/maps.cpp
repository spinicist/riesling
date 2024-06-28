#include "types.hpp"

#include "algo/stats.hpp"
#include "fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/fft.hpp"
#include "op/pad.hpp"
#include "parse_args.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_sense_maps(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file"), oname(parser, "FILE", "Output HD5 file"),
    tname(parser, "FILE", "Recon HD5 file");

  args::ValueFlag<float>                         osamp(parser, "O", "Grid oversampling factor (2)", {"osamp"}, 2.f);
  args::ValueFlag<Eigen::Array3f, Array3fReader> fov(parser, "SENSE-FOV", "SENSE FOV (default header FOV)", {"sense-fov"},
                                                     Eigen::Array3f::Zero());

  ParseCommand(parser, iname, oname);

  HD5::Reader reader(tname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);

  HD5::Reader kreader(iname.Get());
  Cx5 const   kernels = kreader.readTensor<Cx5>();
  Cx5 const   maps = SENSE::KernelsToMaps(kernels, traj.matrix(osamp.Get()), traj.matrixForFOV(fov.Get()));
  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Data, maps.dimensions(), maps.data(), HD5::Dims::SENSE);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
