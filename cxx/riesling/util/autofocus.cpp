#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/patches.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

auto Focus(Cx5 const &x) -> Cx5
{
  Re1 const f = x.imag().abs().sum(Sz4{1, 2, 3, 4});
  Cx5 const y = x.slice(Sz5{I0(f.argmin())(), 0, 0, 0, 0}, AddFront(LastN<4>(x.dimensions()), 1));
  return y;
}

void main_autofocus(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<Index>        patch(parser, "P", "Patch size", {"patch", 'p'}, 5);
  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  Cx5 const   in = reader.readTensor<Cx5>();
  Index const nX = in.dimension(1);
  Index const nY = in.dimension(2);
  Index const nZ = in.dimension(3);
  Index const nT = in.dimension(4);
  Cx5         out(1, nX, nY, nZ, nT);
  Cx5Map      omap(out.data(), out.dimensions());
  auto const &all_start = Log::Now();
  Patches(patch.Get(), 1, false, Focus, in, omap);
  Log::Print(cmd, "All Volumes: {}", Log::ToNow(all_start));

  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info));
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Images);
  Log::Print(cmd, "Finished");
}
