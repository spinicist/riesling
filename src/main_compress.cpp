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

  parser.Parse();
  Log log(verbose);
  log.info(FMT_STRING("Starting operation: {}"), parser.GetCommand().Name());

  RadialReader reader(fname.Get(), log);
  RadialInfo const in_info = reader.info();
  Cx4 in_ks = in_info.radialSeries();
  reader.readData(in_ks);

  Compressor compressor(in_ks.chip(in_info.volumes - 1, 3), cc.Get(), log);
  RadialInfo out_info = in_info;
  out_info.channels = compressor.out_channels();
  Cx4 out_ks = out_info.radialSeries();
  compressor.compress(in_ks, out_ks);

  auto const ofile = OutName(fname, oname, "compressed", "h5");
  RadialWriter writer(ofile, log);
  writer.writeInfo(out_info);
  writer.writeTrajectory(reader.readTrajectory());
  for (long iv = 0; iv < out_info.volumes; iv++) {
    writer.writeData(iv, out_ks.chip(iv, 3));
  }

  fmt::print("{}\n", ofile);
  return EXIT_SUCCESS;
}
