#include "types.hpp"

#include "func/dict.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/pad.hpp"
#include "op/wavelets.hpp"
#include "inputs.hpp"
#include "prox/entropy.hpp"
#include "prox/l1-wavelets.hpp"
#include "prox/llr.hpp"
#include "threads.hpp"

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

  args::ValueFlag<float>            wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"});
  args::ValueFlag<Sz4, SzReader<4>> waveDims(parser, "W", "Wavelet denoising levels", {"wavelet-dims"}, Sz4{0, 1, 1, 1});
  args::ValueFlag<Index>            waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6);

  ParseCommand(parser);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5 const   images = input.readTensor<Cx5>() * Cx(scale.Get());
  Cx5         output(images.dimensions());

  using Map = Proxs::Prox<Cx>::Map;
  using CMap = Proxs::Prox<Cx>::CMap;

  Sz4 const                        dims = FirstN<4>(images.dimensions());
  Index const                      nvox = Product(dims);
  std::shared_ptr<Proxs::Prox<Cx>> prox;
  if (wavelets) {
    prox = std::make_shared<Proxs::L1Wavelets>(wavelets.Get(), dims, waveWidth.Get(), waveDims.Get());
  } else if (llr) {
    prox = std::make_shared<Proxs::LLR>(llr.Get(), llrPatch.Get(), llrWin.Get(), llrShift, dims);
  } else if (l1) {
    prox = std::make_shared<Proxs::L1>(l1.Get(), nvox);
  } else if (nmrent) {
    prox = std::make_shared<Proxs::Entropy>(nmrent.Get(), nvox);
  } else {
    throw args::Error("Must specify at least one regularization method");
  }
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    CMap im(&images(0, 0, 0, 0, iv), nvox);
    Map  om(&output(0, 0, 0, 0, iv), nvox);
    prox->apply(1.f, im, om);
    om = om / scale.Get();
  }
  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Image);
  rl::Log::Print("Finished {}", parser.GetCommand().Name());
}
