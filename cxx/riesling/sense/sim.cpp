#include "io/hd5.hpp"
#include "log.hpp"
#include "inputs.hpp"
#include "sense/coils.hpp"
#include "sense/sense.hpp"
#include "tensors.hpp"
#include "types.hpp"

#include <filesystem>

using namespace rl;

void main_sense_sim(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "FILE", "Filename to write SENSE maps to");

  args::ValueFlag<float> voxel_size(parser, "V", "Voxel size in mm (default 2)", {'v', "vox-size"}, 2.f);
  args::ValueFlag<Index> matrix(parser, "M", "Matrix size (default 128)", {'m', "matrix"}, 128);

  args::ValueFlag<Index> nchan(parser, "C", "Number of channels (8)", {'c', "channels"}, 8);
  args::ValueFlag<Index> coil_rings(parser, "R", "Number of rings in coil (default 1)", {"rings"}, 1);
  args::ValueFlag<float> coil_r(parser, "R", "Radius of the coil in mm (default 192.f)", {"coil-radius"}, 192.f);

  ParseCommand(parser, oname);

  Sz3 shape{matrix.Get(), matrix.Get(), matrix.Get()};
  Cx4 sense =
    birdcage(shape, Eigen::Array3f::Constant(voxel_size.Get()), nchan.Get(), coil_rings.Get(), coil_r.Get(), coil_r.Get());

  // Normalize
  sense /= DimDot<0>(sense, sense).sqrt().reshape(AddFront(shape, 1)).broadcast(Sz4{nchan.Get(), 1, 1, 1});

  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Data, AddFront(sense.dimensions(), 1), sense.data(), HD5::Dims::SENSE);
}
