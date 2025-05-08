#include "rl/basis/basis.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/types.hpp"

#include "args.hpp"

void main_navs(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "I", "File for motion correction");
  args::Positional<std::string> oname(parser, "O", "Name for the basis file");
  
  args::ValueFlag<Index> tracesPerNav(parser, "T", "Traces per navigator", {'t', "traces-per-nav"}, 64);
  args::ValueFlag<Index> tracesPerWindow(parser, "T", "Traces per window", {'w', "traces-per-window"});
  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!oname) { throw args::Error("No output filename specified"); }

  Index const TPN = tracesPerNav.Get();
  Index const TPW = tracesPerWindow ? tracesPerWindow.Get() : TPN;
  if (TPW < TPN) { throw rl::Log::Failure(cmd, "Window size {} was less than frame size {}", TPW, TPN); }
  rl::HD5::Reader ifile(iname.Get());
  auto const ishape = ifile.dimensions();
  Index const nT = ishape[2];
  Index const nNav = nT / TPN;
  
  rl::Log::Print(cmd, "Navigators {} Traces per navigator {} per window", nNav, TPN, TPW);
  rl::Re3 basis(nNav, 1, nT);
  basis.setZero();
  for (Index in = 0; in < nNav; in++) {
    Index const mid = (in + 0.5f) * TPN;
    Index const start = std::clamp((Index)(mid - 0.5f * TPW), 0L, nT - TPW);
    for (Index it = 0; it < TPW; it++) {
      basis(in, 0, start + it) = 1.;
    }
  }
  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, basis.dimensions(), basis.data(), rl::HD5::Dims::Basis);
}
