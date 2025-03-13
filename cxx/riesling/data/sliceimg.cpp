#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_slice_img(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Sz2, SzReader<2>> ia(parser, "I", "i start,size", {'i', "i"});
  args::ValueFlag<Sz2, SzReader<2>> ja(parser, "J", "j start,size", {'j', "j"});
  args::ValueFlag<Sz2, SzReader<2>> ka(parser, "K", "k start,size", {'k', "k"});
  args::ValueFlag<Sz2, SzReader<2>> ba(parser, "B", "b start,size", {'b', "b"});
  args::ValueFlag<Sz2, SzReader<2>> ta(parser, "T", "t start,size", {'t', "t"});

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  auto const  info = reader.readInfo();

  if (reader.order() != 5) { throw Log::Failure(cmd, "Dataset does not appear to be images with 5 dimensions"); }
  auto const shape = reader.dimensions();

  Sz2 const i = ia ? ia.Get() : Sz2{0, shape[0]};
  Sz2 const j = ja ? ja.Get() : Sz2{0, shape[1]};
  Sz2 const k = ka ? ka.Get() : Sz2{0, shape[2]};
  Sz2 const b = ba ? ba.Get() : Sz2{0, shape[3]};
  Sz2 const t = ta ? ta.Get() : Sz2{0, shape[4]};

  Log::Print(cmd, "Selected slice {}:{}, {}:{}, {}:{}, {}:{}, {}:{}", i[0], i[1], j[0], j[1], k[0], k[1], b[0], b[1], t[0],
             t[1]);

  auto imgs = reader.readTensor<Cx5>();
  imgs = Cx5(imgs.slice(Sz5{i[0], j[0], k[0], b[0], t[0]}, Sz5{i[1], j[1], k[1], b[1], t[1]}));

  HD5::Writer writer(oname.Get());
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, imgs.dimensions(), imgs.data(), HD5::Dims::Images);
  Log::Print(cmd, "Finished");
}
