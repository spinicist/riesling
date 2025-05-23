#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/op/sense.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_op_sense(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::Positional<std::string> sname(parser, "FILE", "SENSE maps HD5 file");
  args::Flag                    fwd(parser, "F", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset name (image/channels)", {'d', "dset"});
  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader ireader(iname.Get());
  if (!sname) { throw Log::Failure(cmd, "No input SENSE map file specified"); }
  HD5::Reader      sreader(sname.Get());
  auto const       maps = sreader.readTensor<Cx5>();
  auto const       shape = FirstN<3>(maps.dimensions());
  auto const       nC = maps.dimension(3);
  Trajectory const traj(ireader, ireader.readInfo().voxel_size);
  HD5::Writer      writer(oname.Get());
  writer.writeInfo(ireader.readInfo());
  traj.write(writer);

  if (fwd) {
    auto const    images = ireader.readTensor<Cx5>();
    auto const    nB = images.dimension(3);
    auto const    nT = images.dimension(4);
    TOps::SENSEOp sense(maps, nB);
    Cx6           channels(AddBack(shape, nC, nB, nT));
    for (auto it = 0; it < nT; it++) {
      sense.forward(CChipMap(images, it), ChipMap(channels, it));
    }
    writer.writeTensor(HD5::Keys::Data, channels.dimensions(), channels.data(), HD5::Dims::Channels);
  } else {
    auto const    channels = ireader.readTensor<Cx6>();
    auto const    nB = channels.dimension(4);
    auto const    nT = channels.dimension(5);
    TOps::SENSEOp sense(maps, nB);
    Cx5           images(AddBack(shape, nB, nT));
    for (auto it = 0; it < nT; it++) {
      sense.adjoint(CChipMap(channels, it), ChipMap(images, it));
    }
    writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Images);
  }
  Log::Print(cmd, "Finished");
}
