#include "args.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/trajectory.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_apply(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> tname(parser, "FILE", "Transforms HD5 file", {"transforms"});
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<Index>        tpnav(parser, "T", "Traces per navigator (1024)", {"traces-per-nav"}, 1024);
  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  Info const  info = reader.readStruct<Info>(HD5::Keys::Info);
  Trajectory  traj(reader, info.voxel_size);
  Cx5         ks = reader.readTensor<Cx5>();

  HD5::Reader tfile(tname.Get());
  auto const  ts = tfile.list();
  for (auto const &t : ts) {
    auto const tfm = tfile.readStruct<rl::Transform>(t);
    auto const inav = std::stol(t);
    Log::Print(cmd, "Moving navigator {}", t);
    traj.moveInFOV(tfm.R, info.direction.inverse() * tfm.δ, inav * tpnav.Get(), tpnav.Get(), ks);
  }
  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  traj.write(writer);
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
  Log::Print(cmd, "Finished");
}
