#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/slice.hpp"

using namespace rl;

void main_slice_nc(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  SzFlag<3>              channel(parser, "CHANNEL", "Channel start,size,stride", {"channel"}, Sz3{0, 0, 1});
  SzFlag<3>              sample(parser, "SAMPLE", "Sample start,size,stride", {"sample"}, Sz3{0, 0, 1});
  SzFlag<3>              trace(parser, "TRACE", "Trace start,size,stride", {"trace"}, Sz3{0, 0, 1});
  SzFlag<3>              segment(parser, "SEG", "Segment start,size,stride", {"segment"}, Sz3{0, 0, 1});
  SzFlag<3>              slab(parser, "SLAB", "Slab start,size,stride", {"slab"}, Sz3{0, 0, 1});
  SzFlag<3>              time(parser, "TIME", "Time start,size,stride", {"time"}, Sz3{0, 0, 1});
  args::ValueFlag<Index> tps(parser, "SEG", "Traces per segment", {"tps"}, 0);

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  auto const  info = reader.readStruct<Info>(HD5::Keys::Info);

  if (reader.order() != 5) { throw Log::Failure(cmd, "Dataset does not appear to be non-cartesian with 5 dimensions"); }
  Cx5        ks = reader.readTensor<Cx5>();
  Trajectory traj(reader, info.voxel_size);
  Re3        tp = traj.points();
  auto const sliced =
    SliceNC(channel.Get(), sample.Get(), trace.Get(), slab.Get(), time.Get(), tps.Get(), segment.Get(), ks, tp);
  Trajectory  newTraj(sliced.tp, traj.matrix(), traj.voxelSize());
  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  newTraj.write(writer);
  writer.writeTensor(HD5::Keys::Data, sliced.ks.dimensions(), sliced.ks.data(), HD5::Dims::Noncartesian);
  Log::Print(cmd, "Finished");
}
