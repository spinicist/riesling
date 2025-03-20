#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_real(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Index> refb(parser, "B", "Reference basis (0)", {'b', "b"}, 0);
  args::ValueFlag<Index> reft(parser, "T", "Reference time (0)", {'t', "t"}, 0);

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  auto const  info = reader.readInfo();

  if (reader.order() != 5) { throw Log::Failure(cmd, "Dataset does not appear to be images with 5 dimensions"); }
  auto       imgs = reader.readTensor<Cx5>();
  auto const ishape = imgs.dimensions();
  auto const rshape = FirstN<3>(ishape);

  Cx3 ref = imgs.chip<4>(reft.Get()).chip<3>(refb.Get());
  Cx3 phs = ref / ref.abs();

  imgs = imgs / phs.reshape(AddBack(rshape, 1, 1)).broadcast(AddFront(LastN<2>(ishape), 1, 1, 1));
  Re5 const real = imgs.real();

  HD5::Writer writer(oname.Get());
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, ishape, real.data(), HD5::Dims::Images);
  Log::Print(cmd, "Finished");
}
