#include "args.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/trajectory.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_move(args::Subparser &parser)
{
  args::Positional<std::string>                    iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string>                    oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<Eigen::Matrix3f, Matrix3fReader> R(parser, "R", "Rotation matrix", {'R', "R"}, Eigen::Matrix3f::Identity());
  args::ValueFlag<Eigen::Vector3f, Vector3fReader> shift(parser, "S", "Shift in mm", {"shift"});
  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  Cx5         ks = reader.readTensor<Cx5>();

  traj.moveInFOV(R.Get(), info.direction.inverse() * shift.Get(), ks);

  HD5::Writer writer(oname.Get());
  writer.writeInfo(info);
  traj.write(writer);
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
  Log::Print(cmd, "Finished");
}
