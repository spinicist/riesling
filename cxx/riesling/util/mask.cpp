#include "inputs.hpp"
#include "outputs.hpp"
#include "rl/algo/otsu.hpp"
#include "rl/log.hpp"

using namespace rl;

void main_mask(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<float> thresh(parser, "T", "Use simple threshold instead of Otsu", {'t', "thresh"});
  args::Flag k(parser, "K", "Crop bottom half in k direction", {'k'});

  ParseCommand(parser, iname, oname);
  auto const cmd = parser.GetCommand().Name();
  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader ifile(iname.Get());
  auto const info = ifile.readInfo();
  Re5 const   in = ifile.readTensor<Cx5>().abs();
  float       t = 0.f;
  if (thresh) {
    t = thresh.Get();
  } else {
    auto const o = Otsu(CollapseToConstVector(in));
    t = o.thresh;
  }
  Re5 out = (in > t).select(in.constant(1.f), in.constant(0.f));

  if (k) {
    Sz5 st, sz = out.dimensions();
    sz[2] = sz[2] / 2;
    out.slice(st, sz).setZero();
  }

  HD5::Writer ofile(oname.Get());
  ofile.writeInfo(info);
  ofile.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Image);
  Log::Print(cmd, "Finished");
}
