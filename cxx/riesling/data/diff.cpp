#include "types.hpp"

#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "sys/threads.hpp"
#include "tensors.hpp"

using namespace rl;

void main_diff(args::Subparser &parser)
{
  args::Positional<std::string> aname(parser, "FILE", "Input HD5 file A");
  args::Positional<std::string> bname(parser, "FILE", "Input HD5 file B");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output (B - A) file");
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset name", {'d', "data"}, "data");
  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!aname) { throw args::Error("No file A specified"); }
  if (!bname) { throw args::Error("No file B specified"); }
  if (!oname) { throw args::Error("No output file specified"); }
  HD5::Reader readerA(aname.Get());
  HD5::Reader readerB(bname.Get());

  HD5::Writer writer(oname.Get());
  writer.writeInfo(readerA.readInfo());

  auto const orderA = readerA.order(dset.Get());
  auto const orderB = readerB.order(dset.Get());

  if (orderA != orderB) {
    Log::Fail(cmd, "Dataset {} in file {} had order {} but in file {} it was {}", dset.Get(), aname.Get(), orderA, bname.Get(),
              orderB);
  }

  auto const shapeA = readerA.dimensions(dset.Get());
  auto const shapeB = readerB.dimensions(dset.Get());

  if (shapeA != shapeB) {
    Log::Fail(cmd, "Dataset {} in file {} had shape {} but in file {} it was {}", dset.Get(), aname.Get(), shapeA, bname.Get(),
              shapeB);
  }

  switch (orderA) {
  case 4: {
    Cx4 const A = readerA.readTensor<Cx4>(dset.Get());
    Cx4 const B = readerB.readTensor<Cx4>(dset.Get());
    Cx4 const diff = B - A;
    writer.writeTensor(dset.Get(), diff.dimensions(), diff.data(), readerA.dimensionNames<4>());
  } break;
  case 5: {
    Cx5 const A = readerA.readTensor<Cx5>(dset.Get());
    Cx5 const B = readerB.readTensor<Cx5>(dset.Get());
    Cx5 const diff = B - A;
    writer.writeTensor(dset.Get(), diff.dimensions(), diff.data(), readerA.dimensionNames<5>());
  } break;
  case 6: {
    Cx6 const A = readerA.readTensor<Cx6>(dset.Get());
    Cx6 const B = readerB.readTensor<Cx6>(dset.Get());
    Cx6 const diff = B - A;
    writer.writeTensor(dset.Get(), diff.dimensions(), diff.data(), readerA.dimensionNames<6>());
  } break;
  default: Log::Fail(cmd, "Data had order {}, I'm lazy", orderA);
  }
  Log::Print(cmd, "Finished");
}
