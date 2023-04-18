
#include "coils.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sense.hpp"
#include "types.hpp"
#include <filesystem>

using namespace rl;

int main_sense_sim(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Filename to write SENSE maps to");

  args::ValueFlag<float> voxel_size(parser, "V", "Voxel size in mm (default 2)", {'v', "vox-size"}, 2.f);
  args::ValueFlag<Index> matrix(parser, "M", "Matrix size (default 128)", {'m', "matrix"}, 128);

  args::ValueFlag<Index> nchan(parser, "C", "Number of channels (8)", {'c', "channels"}, 8);
  args::ValueFlag<Index> coil_rings(parser, "R", "Number of rings in coil (default 1)", {"rings"}, 1);
  args::ValueFlag<float> coil_r(parser, "R", "Radius of the coil in mm (default 192.f)", {"coil-radius"}, 192.f);

  ParseCommand(parser, iname);

  Cx4 sense = birdcage(
    Sz3{matrix.Get(), matrix.Get(), matrix.Get()},
    Eigen::Array3f::Constant(voxel_size.Get()),
    nchan.Get(),
    coil_rings.Get(),
    coil_r.Get(),
    coil_r.Get());

  auto const fname = OutName("", iname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(HD5::Keys::SENSE, sense.dimensions(), sense.data());

  return EXIT_SUCCESS;
}
