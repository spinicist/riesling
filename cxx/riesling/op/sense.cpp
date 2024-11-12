#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/sense.hpp"
#include "inputs.hpp"
#include "sys/threads.hpp"

using namespace rl;

void main_op_sense(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::Positional<std::string> sname(parser, "FILE", "SENSE maps HD5 file");
  args::Flag                    fwd(parser, "F", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset name (image/channels)", {'d', "dset"});
  ParseCommand(parser, iname);
  auto const cmd = parser.GetCommand().Name();
  HD5::Reader ireader(iname.Get());
  if (!sname) { throw Log::Failure(cmd, "No input SENSE map file specified"); }
  HD5::Reader sreader(sname.Get());
  auto const  maps = sreader.readTensor<Cx5>();

  Trajectory const traj(ireader, ireader.readInfo().voxel_size);
  HD5::Writer      writer(oname.Get());
  writer.writeInfo(ireader.readInfo());
  traj.write(writer);

  if (fwd) {
    auto const images = ireader.readTensor<Cx5>();
    if (LastN<3>(maps.dimensions()) != MidN<1, 3>(images.dimensions())) {
      throw Log::Failure(cmd, "Image dimensions {} did not match SENSE maps {}", images.dimensions(), maps.dimensions());
    }
    TOps::SENSE sense(maps, false, 1);
    Cx6         channels(AddBack(maps.dimensions(), images.dimension(4)));
    for (auto ii = 0; ii < images.dimension(4); ii++) {
      auto const temp = sense.forward(CChipMap(images, ii));
      channels.chip<5>(ii).device(Threads::TensorDevice()) = temp;
    }
    writer.writeTensor(HD5::Keys::Data, channels.dimensions(), channels.data(), HD5::Dims::Channels);
  } else {
    auto const channels = ireader.readTensor<Cx6>();
    if (maps.dimensions() != FirstN<5>(channels.dimensions())) {
      throw Log::Failure(cmd, "Channel dimensions {} did not match SENSE maps {}", channels.dimensions(), maps.dimensions());
    }
    TOps::SENSE sense(maps, false, 1);
    Cx5         images(LastN<5>(channels.dimensions()));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      images.chip<4>(ii).device(Threads::TensorDevice()) = sense.adjoint(CChipMap(channels, ii));
    }
    writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Image);
  }
  Log::Print(cmd, "Finished");
}
