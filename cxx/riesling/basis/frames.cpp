#include "types.hpp"

#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "inputs.hpp"

void main_frames(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> tracesPerFrame(parser, "T", "Traces per frame", {"tpf"});
  args::ValueFlag<Index> framesPerRep(parser, "T", "Frames per repitition", {"fpr"});
  args::ValueFlag<Index> reps(parser, "R", "Repetitions", {"reps"});
  args::ValueFlag<Index> startFrame(parser, "F", "Start frame", {"start"});
  args::ValueFlag<Index> incFrame(parser, "F", "Frame increment", {"inc"}, 1);
  args::ValueFlag<Index> retain(parser, "R", "Frames to retain", {"retain"});
  ParseCommand(parser);
  if (!oname) { throw args::Error("No output filename specified"); }

  Index const nF = reps ? reps.Get() : framesPerRep.Get();
  Index const nT = tracesPerFrame.Get() * framesPerRep.Get() * (reps ? reps.Get() : 1);
  Index       index = startFrame ? startFrame.Get() * tracesPerFrame.Get() : 0;
  rl::Log::Print("Frames {} Traces {}", nF, nT);
  rl::Re3 basis(nF, 1, nT);
  basis.setZero();
  for (Index ifr = 0; ifr < nF; ifr++) {
    for (Index it = 0; it < tracesPerFrame.Get(); it++) {
      if (index >= nT) { rl::Log::Fail("Trace index {} exceeded maximum {}", index, nT); }
      basis(ifr, 0, index++) = 1.;
    }
    if (incFrame) { index += incFrame.Get() * tracesPerFrame.Get(); }
  }

  if (retain) {
    basis = rl::Re3(basis.slice(rl::Sz3{0, 0, 0}, rl::Sz3{retain.Get(), 1, nT}));
  }

  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, basis.dimensions(), basis.data(), rl::HD5::Dims::Basis);
}
