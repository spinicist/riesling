#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"

using namespace rl;

int main_sdc(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  args::ValueFlag<std::string> sdcType(parser, "SDC", "SDC type: 'pipe', 'radial'", {"sdc"}, "pipe");
  args::ValueFlag<Index>       lores(parser, "L", "Number of lo-res traces for radial", {'l', "lores"}, 0);
  args::ValueFlag<Index>       gap(parser, "G", "Read-gap for radial", {'g', "gap"}, 0);
  args::ValueFlag<Index>       its(parser, "N", "Maximum number of iterations (40)", {"max-its", 'n'}, 40);
  ParseCommand(parser, coreOpts.iname);
  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));

  Re2 dc;
  if (sdcType.Get() == "pipe") {
    switch (traj.nDims()) {
    case 2: dc = SDC::Pipe<2>(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), its.Get()); break;
    case 3: dc = SDC::Pipe<3>(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), its.Get()); break;
    }
  } else if (sdcType.Get() == "radial") {
    dc = SDC::Radial(traj, lores.Get(), gap.Get());
  } else {
    Log::Fail("Uknown SDC method: {}", sdcType.Get());
  }
  HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "sdc", "h5"));
  writer.writeTensor(HD5::Keys::SDC, dc.dimensions(), dc.data());
  return EXIT_SUCCESS;
}
