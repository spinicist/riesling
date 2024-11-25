#include "inputs.hpp"

#include "rl/func/dict.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/op/wavelets.hpp"
#include "rl/prox/entropy.hpp"
#include "rl/prox/l1-wavelets.hpp"
#include "rl/prox/llr.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_prox(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<float>        scale(parser, "S", "Scale data before applying prox", {'s', "scale"}, 1);
  args::ValueFlag<float>        l1(parser, "L1", "Simple L1 regularization", {"l1"});
  args::ValueFlag<float>        nmrent(parser, "E", "NMR Entropy", {"nmrent"});

  args::ValueFlag<float> llr(parser, "L", "LLR regularization", {"llr"});
  args::ValueFlag<Index> llrPatch(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> llrWin(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);
  args::Flag             llrShift(parser, "S", "Enable random LLR shifting", {"llr-shift"});

  args::ValueFlag<float> wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"});
  VectorFlag<Index>      waveDims(parser, "W", "Wavelet denoising levels", {"wavelet-dims"}, std::vector<Index>{1, 2, 3});
  args::ValueFlag<Index> waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6);

  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5 const   images = input.readTensor<Cx5>() * Cx(scale.Get());
  Cx5         output(images.dimensions());

  using Map = Proxs::Prox<Cx>::Map;
  using CMap = Proxs::Prox<Cx>::CMap;
  CMap im(images.data(), images.size());
  Map  om(output.data(), output.size());

  Sz5 const                        shape = images.dimensions();
  Index const                      nvox = Product(shape);
  std::shared_ptr<Proxs::Prox<Cx>> prox;
  if (wavelets) {
    prox = std::make_shared<Proxs::L1Wavelets>(wavelets.Get(), shape, waveWidth.Get(), waveDims.Get());
  } else if (llr) {
    prox = std::make_shared<Proxs::LLR>(llr.Get(), llrPatch.Get(), llrWin.Get(), llrShift, shape);
  } else if (l1) {
    prox = std::make_shared<Proxs::L1>(l1.Get(), nvox);
  } else if (nmrent) {
    prox = std::make_shared<Proxs::Entropy>(nmrent.Get(), nvox);
  } else {
    throw args::Error("Must specify at least one regularization method");
  }
  prox->apply(1.f, im, om);
  output = output / output.constant(scale.Get());

  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Image);
  rl::Log::Print(cmd, "Finished");
}
