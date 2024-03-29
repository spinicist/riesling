#include "types.hpp"

#include "io/reader.hpp"
#include "io/writer.hpp"
#include "log.hpp"
#include "parse_args.hpp"

int main_ipop_combine(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "OUTPUT", "Input filename");

  args::Flag wf(parser, "W", "Input is Water/Fat, output will be IP/OP", {"wf"});
  ParseCommand(parser, iname);

  rl::HD5::Reader ifile(iname.Get());

  rl::Cx5 const input = ifile.readTensor<rl::Cx5>(rl::HD5::Keys::Image);
  rl::Cx5       output(input.dimensions());

  std::string suffix;
  if (wf) {
    output.chip<0>(0) = 0.5f * (input.chip<0>(0) + input.chip<0>(1));
    output.chip<0>(1) = 0.5f * (input.chip<0>(0) - input.chip<0>(1));
    suffix = "ipop";
  } else {
    output.chip<0>(0) = input.chip<0>(0) + input.chip<0>(1);
    output.chip<0>(1) = input.chip<0>(0) - input.chip<0>(1);
    suffix = "wf";
  }

  rl::HD5::Writer writer(OutName(iname.Get(), "", suffix));
  writer.writeTensor(rl::HD5::Keys::Image, output.dimensions(), output.data(), rl::HD5::Dims::Image);
  return EXIT_SUCCESS;
}
