#include "inputs.hpp"

#include "rl/filter.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_filter(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "ifile HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  ArrayFlag<float, 3>           filter(parser, "F", "Filter start,end,height", {'f', "filter"});
  ParseCommand(parser, iname, oname);
  auto const cmd = parser.GetCommand().Name();

  HD5::Reader ifile(iname.Get());
  Info        info = ifile.readStruct<Info>(HD5::Keys::Info);
  Trajectory  traj(ifile, info.voxel_size);
  auto        ks = ifile.readTensor<Cx5>();
  float const M = *std::max_element(traj.matrix().cbegin(), traj.matrix().cend()) / 2.f;
  auto const  f = filter.Get();
  NoncartesianTukey(f[0] * M, f[1] * M, f[2], traj.points(), ks);
  HD5::Writer ofile(oname.Get());
  traj.write(ofile);
  ofile.writeStruct(HD5::Keys::Info, info);
  ofile.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
  Log::Print(cmd, "Finished");
}
