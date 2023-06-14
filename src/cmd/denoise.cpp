#include "types.hpp"

#include "algo/decomp.hpp"
#include "fft/fft.hpp"
#include "filter.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "patches.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

int main_denoise(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});

  args::Flag llr(parser, "L", "LLR denoising", {"llr"});
  args::Flag llrFft(parser, "L", "LLR denoising in the Fourier domain", {"llr-fft"});
  args::ValueFlag<Index> llrPatch(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> llrWin(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);
  args::ValueFlag<float> λ(parser, "λ", "Threshold", {"lambda"}, 1.f);

  ParseCommand(parser);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());

  Cx5 images = input.readTensor<Cx5>(HD5::Keys::Image);
  auto const fft = llrFft ? FFT::Make<4, 3>(FirstN<4>(images.dimensions())) : nullptr;

  auto hardLLR = [λ = λ.Get()](Cx4 const &xp) {
    Eigen::MatrixXcf patch = CollapseToMatrix(xp);
    auto const svd = SVD<Cx>(patch, true, false);
    // Soft-threhold svals
    Eigen::VectorXf const s = (svd.vals.abs() > λ).select(svd.vals, 0.f);
    patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
    Cx4 yp = Tensorfy(patch, xp.dimensions());
    return yp;
  };

  Cx4 img(FirstN<4>(images.dimensions()));
  Cx4 out(FirstN<4>(images.dimensions()));
  Eigen::TensorMap<Cx4> outmap(out.data(), out.dimensions());
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    img = images.chip<4>(iv);
    if (fft) { fft->forward(img); }
    Patches(llrPatch.Get(), llrWin.Get(), hardLLR, ConstMap(img), outmap);
    if (fft) { fft->reverse(out); }
    images.chip<4>(iv) = out;
  }
  auto const fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Image, images.dimensions(), images.data());
  return EXIT_SUCCESS;
}
