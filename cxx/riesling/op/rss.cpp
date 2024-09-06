#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "inputs.hpp"
#include "tensors.hpp"
#include "sys/threads.hpp"

using namespace rl;

void main_rss(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string>  oname(parser, "OUTPUT", "Output file name");
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());

  HD5::Writer writer(oname.Get());
  writer.writeInfo(reader.readInfo());

  auto const order = reader.order();

  switch (order) {
  case 5: {
    Cx5 const in = reader.readTensor<Cx5>();
    Cx4 const out = DimDot<1>(in, in).sqrt();
    writer.writeTensor(HD5::Keys::Data, AddFront(out.dimensions(), 1), out.data(), reader.dimensionNames<5>());
  } break;
  case 6: {
    Cx6 const in = reader.readTensor<Cx6>();
    Cx5 const out = DimDot<1>(in, in).sqrt();
    writer.writeTensor(HD5::Keys::Data, AddFront(out.dimensions(), 1), out.data(), reader.dimensionNames<6>());
  } break;
  default: Log::Fail("Data had order {}", order);
  }
}
