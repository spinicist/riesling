#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/make_grid.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense.hpp"

using namespace rl;

int main_sense_calib(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SENSE::Opts senseOpts(parser);
  SDC::Opts sdcOpts(parser, "pipe");

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Cx4 sense = SENSE::SelfCalibration(senseOpts, coreOpts, traj, reader);
  auto const fname = OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(HD5::Keys::SENSE, sense.dimensions(), sense.data());

  return EXIT_SUCCESS;
}
