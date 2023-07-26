#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/make_grid.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

using namespace rl;

int main_sense_calib(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  SENSE::Opts senseOpts(parser);
  SDC::Opts   sdcOpts(parser, "pipe");

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto        noncart = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Cx4         sense = SENSE::Choose(senseOpts, coreOpts, traj, noncart);
  auto const  fname = OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(HD5::Keys::SENSE, sense.dimensions(), sense.data());

  return EXIT_SUCCESS;
}
