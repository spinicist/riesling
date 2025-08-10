#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/sense/sense.hpp"

using namespace rl;

void main_sense_maps(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input HD5 file"),
    tname(parser, "TARGET", "Target HD5 file for reconstruction"), oname(parser, "OUTPUT", "Output HD5 file");

  args::ValueFlag<float> osamp(parser, "O", "Grid oversampling factor (1.3)", {"osamp"}, 1.3f);
  ArrayFlag<float, 3>    fov(parser, "FOV", "Grid FoV in mm (x,y,z)", {"fov"}, Eigen::Array3f::Zero());
  args::ValueFlag<Index> kW(parser, "W", "Turn maps into kernels with specified width", {"sense-width"});
  args::MapFlag<std::string, rl::SENSE::Normalization> renorm(parser, "N", "SENSE Renormalization (RSS/none)", {"sense-renorm"},
                                                              SENSE::NormMap, SENSE::Normalization::RSS);

  ParseCommand(parser, iname, oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader treader(tname.Get());
  Trajectory  traj(treader, treader.readStruct<Info>(HD5::Keys::Info).voxel_size);

  HD5::Reader reader(iname.Get());
  HD5::Writer writer(oname.Get());
  if (kW) {
    Cx5 const maps = reader.readTensor<Cx5>();
    Cx5 const kernels = SENSE::MapsToKernels(maps, Sz3{kW.Get(), kW.Get(), kW.Get()}, osamp.Get());
    writer.writeTensor(HD5::Keys::Data, kernels.dimensions(), kernels.data(), HD5::Dims::SENSE);
  } else {
    Cx5 const kernels = reader.readTensor<Cx5>();
    Cx5 const maps = SENSE::KernelsToMaps(kernels, traj.matrixForFOV(fov.Get()), osamp.Get(), renorm.Get());
    writer.writeTensor(HD5::Keys::Data, maps.dimensions(), maps.data(), HD5::Dims::SENSE);
  }

  Log::Print(cmd, "Finished");
}
