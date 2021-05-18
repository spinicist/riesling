#include "types.h"

#include "gridder.h"
#include "io_hd5.h"
#include "kernels.h"
#include "log.h"
#include "parse_args.h"

enum class Type
{
  Pipe = 1,
  RadialAnalytic = 2
};
std::unordered_map<std::string, Type> TypeMap{{"pipe", Type::Pipe},
                                              {"radial", Type::RadialAnalytic}};

int main_sdc(args::Subparser &parser)
{
  CORE_RECON_ARGS;
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::MapFlag<std::string, Type> type(
      parser, "Type", "1 - Pipe, 2 - Radial Analytic", {"type"}, TypeMap, Type::Pipe);
  Log log = ParseCommand(parser, fname);
  HD5::Reader reader(fname.Get(), log);
  auto const &info = reader.info();
  auto const trajectory = reader.readTrajectory();

  R2 sdc;
  switch (type.Get()) {
  case Type::Pipe: {
    Kernel *kernel =
        kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (info.type == Info::Type::ThreeD))
           : (Kernel *)new NearestNeighbour();
    Gridder gridder(info, trajectory, osamp.Get(), kernel, fastgrid, log);
    sdc = SDC::Pipe(info, gridder, kernel, log);
  } break;
  case Type::RadialAnalytic: {
    sdc = SDC::Radial(info, trajectory, log);
  } break;
  }
  HD5::Writer writer(OutName(fname, oname, "sdc", "h5"), log);
  writer.writeInfo(info);
  writer.writeSDC(sdc);
  return EXIT_SUCCESS;
}
