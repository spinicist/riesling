#include "types.hpp"

#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "inputs.hpp"

void main_echoes(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> nS(parser, "S", "Total samples", {"samples", 's'});
  args::ValueFlag<Index> nG(parser, "G", "Discard samples in gap", {"gap", 'g'});
  args::ValueFlag<Index> nK(parser, "K", "Samples to keep", {"keep", 'k'});
  args::ValueFlag<Index> nE(parser, "E", "Number of echoes", {"echoes", 'e'});

  ParseCommand(parser);
  if (!oname) { throw args::Error("No output filename specified"); }

  if (nG.Get() < 0) { rl::Log::Fail("Gap was negative"); }
  Index const sz = nK ? nK.Get() : nS.Get();
  if (nG.Get() + sz > nS.Get()) {
    rl::Log::Fail("Gap {} plus samples to keep {} is larger than number of samples {}", nG.Get(), sz, nS.Get());
  }

  Index sampPerEcho = nS.Get() / nE.Get();
  rl::Log::Print("Echoes {} Samples {} Keep {}-{} Samples-per-echo {}", nE.Get(), nS.Get(), nG.Get(), nG.Get() + sz, sampPerEcho);
  rl::Re3 basis(nE.Get(), sz, 1);
  basis.setZero();
  float const scale = std::sqrt(nE.Get());
  for (Index is = 0; is < sz; is++) {
    Index const ind = (nG.Get() + is) / sampPerEcho;
    basis(ind, is, 0) = scale;
  }

  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, basis.dimensions(), basis.data(), rl::HD5::Dims::Basis);
  rl::Log::Print("Finished {}", parser.GetCommand().Name());
}
