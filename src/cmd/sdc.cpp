#include "types.hpp"

#include "sdc.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grid.hpp"
#include "parse_args.hpp"

using namespace rl;

int main_sdc(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  GridOpts                     gridOpts(parser);
  args::ValueFlag<std::string> sdcType(parser, "SDC", "SDC type: 'pipe', 'radial'", {"sdc"}, "pipe");
  args::ValueFlag<Index>       lores(parser, "L", "Number of lo-res traces for radial", {'l', "lores"}, 0);
  args::ValueFlag<Index>       its(parser, "N", "Maximum number of iterations (40)", {"max-its", 'n'}, 40);
  ParseCommand(parser, coreOpts.iname);
  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));

  Re2 dc;
  if (sdcType.Get() == "pipe") {
    switch (traj.nDims()) {
    case 2: dc = SDC::Pipe<2>(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), its.Get()); break;
    case 3: dc = SDC::Pipe<3>(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), its.Get()); break;
    }
  } else if (sdcType.Get() == "radial") {
    dc = SDC::Radial3D(traj, lores.Get());
  } else {
    Log::Fail("Uknown SDC method: {}", sdcType.Get());
  }
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor(HD5::Keys::Weights, dc.dimensions(), dc.data());
  return EXIT_SUCCESS;
}
