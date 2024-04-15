#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "patches.hpp"

using namespace rl;

auto Focus(Cx4 const &x) -> Cx4 {
    Re1 const f = x.imag().abs().sum(Sz3{1, 2, 3});
    Cx4 const y = x.slice(Sz4{I0(f.argmin())(), 0, 0, 0}, Sz4{1, x.dimension(1), x.dimension(2), x.dimension(3)});
    return y;
}

void main_autofocus(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<Index> patch(parser, "P", "Patch size", {"patch", 'p'}, 5);
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());
  Cx5 const in = reader.readTensor<Cx5>();
  Index const nX = in.dimension(1);
  Index const nY = in.dimension(2);
  Index const nZ = in.dimension(3);
  Index const nT = in.dimension(4);
  Cx5 out(1, nX, nY, nZ, nT);
  Cx4 temp(1, nX, nY, nZ);
  Eigen::TensorMap<Cx4> tmap(temp.data(), temp.dimensions());
  auto const &all_start = Log::Now();
  for (Index it = 0; it < nT; it++) {
    Patches(patch.Get(), 1, false, Focus, CChipMap(in, it), tmap);
    out.chip<4>(it) = temp;
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));

  HD5::Writer writer(oname.Get());
  writer.writeInfo(reader.readInfo());
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Image);
}
