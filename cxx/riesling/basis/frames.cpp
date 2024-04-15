#include "types.hpp"

#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"

void main_frames(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> tracesPerFrame(parser, "T", "Traces per frame", {"tpf"});
  args::ValueFlag<Index> framesPerRep(parser, "T", "Frames per repitition", {"fpr"});
  args::ValueFlag<Index> reps(parser, "R", "Repetitions", {"reps"});
  args::ValueFlag<Index> startFrame(parser, "F", "Start frame", {"start"});
  args::ValueFlag<Index> incFrame(parser, "F", "Frame increment", {"inc"}, 1);
  ParseCommand(parser);
  if (!oname) { throw args::Error("No output filename specified"); }

  Index const nF = reps ? reps.Get() : framesPerRep.Get();
  Index const nT = tracesPerFrame.Get() * framesPerRep.Get() * (reps ? reps.Get() : 1);
  Index       index = startFrame ? startFrame.Get() * tracesPerFrame.Get() : 0;
  rl::Log::Print("Frames {} Traces {}", nF, nT);
  rl::Re2 basis(nF, nT);
  basis.setZero();
  for (Index ifr = 0; ifr < nF; ifr++) {
    for (Index it = 0; it < tracesPerFrame.Get(); it++) {
      if (index >= nT) { rl::Log::Fail("Trace index {} exceeded maximum {}", index, nT); }
      basis(ifr, index++) = 1.;
    }
    if (incFrame) { index += incFrame.Get() * tracesPerFrame.Get(); }
  }

  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, rl::Sz3{basis.dimension(0), 1, basis.dimension(1)}, basis.data(), rl::HD5::Dims::Basis);
}
