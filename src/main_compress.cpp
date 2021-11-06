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
  args::ValueFlag<long> cc(parser, "CHANNEL COUNT", "Retain N channels (default 8)", {"cc"}, 8);
  args::ValueFlag<long> ref_vol(
    parser, "REF VOLUME", "Calculate PCA from this volume (default last)", {"vol"});
  args::ValueFlag<long> readSize(
    parser, "READ SIZE", "Number of read-out points to use in PCA (default 16)", {"read"}, 16);
  args::ValueFlag<long> spokeStride(
    parser, "SPOKE STRIDE", "Stride/subsample across spokes (default 4)", {"spokes"}, 4);
  Log log = ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get(), log);
  Info const in_info = reader.readInfo();
  Cx3 ks = reader.noncartesian(ValOrLast(ref_vol, in_info.volumes));
  long const max_ref = in_info.read_points - in_info.read_gap;
  long const nread = (readSize.Get() > max_ref) ? max_ref : readSize.Get();
  log.info(
    FMT_STRING("Using {} read points and {} spokes"), nread, in_info.spokes_hi / spokeStride.Get());
  Cx3 ref = ks.slice(
                Sz3{0, in_info.read_gap, in_info.spokes_lo},
                Sz3{in_info.channels, nread, in_info.spokes_hi})
              .stride(Sz3{1, 1, spokeStride.Get()});
  Compressor compressor(ref, cc.Get(), log);
  Info out_info = in_info;
  out_info.channels = compressor.out_channels();

  Cx4 all_ks = in_info.noncartesianSeries();
  for (long iv = 0; iv < in_info.volumes; iv++) {
    all_ks.chip(iv, 3) = reader.noncartesian(iv);
  }
  Cx4 out_ks = out_info.noncartesianSeries();
  compressor.compress(all_ks, out_ks);

  auto const ofile = OutName(iname.Get(), oname.Get(), "compressed", "h5");
  HD5::Writer writer(ofile, log);
  writer.writeTrajectory(reader.readTrajectory());
  writer.writeNoncartesian(out_ks);
  return EXIT_SUCCESS;
}
