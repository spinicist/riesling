#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_norm(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset name", {'d', "dset"}, "data");
  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!iname) { throw args::Error("No file A specified"); }
  HD5::Reader ifile(iname.Get());
  auto const  order = ifile.order(dset.Get());
  switch (order) {
  case 2: {
    auto const D = ifile.readTensor<Cx2>(dset.Get());
    fmt::print(stdout, "{:4.3E}\n", Norm<true>(D));
  } break;
  case 3: {
    auto const D = ifile.readTensor<Cx3>(dset.Get());
    fmt::print(stdout, "{:4.3E}\n", Norm<true>(D));
  } break;
  case 4: {
    auto const D = ifile.readTensor<Cx4>(dset.Get());
    fmt::print(stdout, "{:4.3E}\n", Norm<true>(D));
  } break;
  case 5: {
    auto const D = ifile.readTensor<Cx5>(dset.Get());
    fmt::print(stdout, "{:4.3E}\n", Norm<true>(D));
  } break;
  case 6: {
    auto const D = ifile.readTensor<Cx6>(dset.Get());
    fmt::print(stdout, "{:4.3E}\n", Norm<true>(D));
  } break;
  default: throw Log::Failure(cmd, "Data had order {}, I'm lazy", order);
  }
  Log::Print(cmd, "Finished");
}
