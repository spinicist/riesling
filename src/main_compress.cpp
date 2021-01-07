#include "compressor.h"
#include "io_hd5.h"
#include "log.h"
#include "parse_args.h"
#include "types.h"

int main_compress(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<long> cc(parser, "CHANNEL COUNT", "Retain N channels (default 8)", {"cc"}, 8);
  args::ValueFlag<long> ref_vol(
      parser, "REF VOLUME", "Calculate PCA from this volume (default last)", {"vol"});
  args::ValueFlag<long> ref_size(
      parser, "REF SIZE", "Number of read-out points to use in PCA (default 16)", {"sz"}, 16);

  parser.Parse();
  Log log(verbose);
  log.info(FMT_STRING("Starting operation: {}"), parser.GetCommand().Name());

  HD5Reader reader(fname.Get(), log);
  Info const in_info = reader.info();
  Cx3 ks = in_info.noncartesianVolume();
  reader.readData(SenseVolume(ref_vol, in_info.volumes), ks);
  long const max_ref = in_info.read_points - in_info.read_gap;
  long const npts = (ref_size.Get() > max_ref) ? max_ref : ref_size.Get();
  Cx3 ref = ks.slice(
      Sz3{0, in_info.read_gap, in_info.spokes_lo}, Sz3{in_info.channels, npts, in_info.spokes_hi});
  Compressor compressor(ref, cc.Get(), log);
  Info out_info = in_info;
  out_info.channels = compressor.out_channels();

  Cx4 all_ks = in_info.noncartesianSeries();
  reader.readData(all_ks);
  Cx4 out_ks = out_info.noncartesianSeries();
  compressor.compress(all_ks, out_ks);

  auto const ofile = OutName(fname, oname, "compressed", "h5");
  HD5Writer writer(ofile, log);
  writer.writeInfo(out_info);
  writer.writeTrajectory(reader.readTrajectory());
  for (long iv = 0; iv < out_info.volumes; iv++) {
    writer.writeData(iv, out_ks.chip(iv, 3));
  }

  fmt::print("{}\n", ofile);
  return EXIT_SUCCESS;
}
