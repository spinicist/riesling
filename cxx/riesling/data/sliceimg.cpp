#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_slice_img(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Sz3, SzReader<3>> ia(parser, "I", "i start,size,stride", {'i', "i"});
  args::ValueFlag<Sz3, SzReader<3>> ja(parser, "J", "j start,size,stride", {'j', "j"});
  args::ValueFlag<Sz3, SzReader<3>> ka(parser, "K", "k start,size,stride", {'k', "k"});
  args::ValueFlag<Sz3, SzReader<3>> ba(parser, "B", "b start,size,stride", {'b', "b"});
  args::ValueFlag<Sz3, SzReader<3>> ta(parser, "T", "t start,size,stride", {'t', "t"});

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  auto const  info = reader.readStruct<Info>(HD5::Keys::Info);

  if (reader.order() != 5) { throw Log::Failure(cmd, "Dataset does not appear to be images with 5 dimensions"); }
  auto const shape = reader.dimensions();

  Sz3 const i = ia ? ia.Get() : Sz3{0, shape[0], 1};
  Sz3 const j = ja ? ja.Get() : Sz3{0, shape[1], 1};
  Sz3 const k = ka ? ka.Get() : Sz3{0, shape[2], 1};
  Sz3 const b = ba ? ba.Get() : Sz3{0, shape[3], 1};
  Sz3 const t = ta ? ta.Get() : Sz3{0, shape[4], 1};

  Log::Print(cmd, "Selected slice {}:{}, {}:{}, {}:{}, {}:{}, {}:{}", i[0], i[1], j[0], j[1], k[0], k[1], b[0], b[1], t[0],
             t[1]);

  auto imgs = reader.readTensor<Cx5>();
  imgs = Cx5(imgs.slice(Sz5{i[0], j[0], k[0], b[0], t[0]}, Sz5{i[1], j[1], k[1], b[1], t[1]}));
  imgs = Cx5(imgs.stride(Sz5{i[2], j[2], k[2], b[2], t[2]}));

  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, imgs.dimensions(), imgs.data(), HD5::Dims::Images);
  Log::Print(cmd, "Finished");
}
