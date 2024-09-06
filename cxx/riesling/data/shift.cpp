#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "tensors.hpp"
#include "types.hpp"

using namespace rl;

void main_shift(args::Subparser &parser)
{
  args::Positional<std::string>                    iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string>                    oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<Eigen::Vector3f, Vector3fReader> shift(parser, "S", "Shift in mm", {"shift"});
  ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  Cx5         ks = reader.readTensor<Cx5>();
  traj.shiftInFOV(shift.Get(), ks);
  HD5::Writer writer(oname.Get());
  writer.writeInfo(info);
  traj.write(writer);
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
