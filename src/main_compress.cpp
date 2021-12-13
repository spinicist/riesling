#include "compressor.h"
#include "io.h"
#include "log.h"
#include "parse_args.h"
#include "sense.h"
#include "types.h"

int main_compress(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<Index> cc(parser, "CHANNEL COUNT", "Retain N channels (8)", {"cc"}, 8);
  args::ValueFlag<Index> ref_vol(parser, "V", "Calculate PCA from volume (last)", {"vol"});
  args::ValueFlag<Index> readStart(parser, "R", "Reference region start (0)", {"read_start"}, 0);
  args::ValueFlag<Index> readSize(parser, "R", "Points for PCA (16)", {"read"}, 16);
  args::ValueFlag<Index> spokeStart(parser, "S", "Ignore first N spokes", {"spoke_start"}, 0);
  args::ValueFlag<Index> spokeStride(parser, "S", "Stride across spokes (4)", {"stride"}, 4);
  Log log = ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get(), log);
  Info const in_info = reader.readInfo();
  Cx3 ks = reader.noncartesian(ValOrLast(ref_vol, in_info.volumes));
  Index const maxRead = in_info.read_points - readStart.Get();
  Index const nread = (readSize.Get() > maxRead) ? maxRead : readSize.Get();
  Index const nspoke = (in_info.spokes - spokeStart.Get()) / spokeStride.Get();
  log.info(
    FMT_STRING("Using {} read points, {} spokes, {} stride"), nread, nspoke, spokeStride.Get());
  Cx3 ref =
    ks.slice(Sz3{0, readStart.Get(), spokeStart.Get()}, Sz3{in_info.channels, nread, nspoke})
      .stride(Sz3{1, 1, spokeStride.Get()});
  Compressor compressor(ref, cc.Get(), log);
  Info out_info = in_info;
  out_info.channels = compressor.out_channels();

  Cx4 all_ks = in_info.noncartesianSeries();
  for (Index iv = 0; iv < in_info.volumes; iv++) {
    all_ks.chip<3>(iv) = reader.noncartesian(iv);
  }
  Cx4 out_ks = out_info.noncartesianSeries();
  compressor.compress(all_ks, out_ks);

  auto const ofile = OutName(iname.Get(), oname.Get(), "compressed", "h5");
  HD5::Writer writer(ofile, log);
  writer.writeTrajectory(reader.readTrajectory());
  writer.writeNoncartesian(out_ks);
  return EXIT_SUCCESS;
}
