#include "types.hpp"

#include "algo/decomp.hpp"
#include "filter.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "patches.hpp"
#include "tensors.hpp"
#include "threads.hpp"

using namespace rl;

void main_denoise(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::Flag             llr(parser, "L", "LLR denoising", {"llr"});
  args::ValueFlag<Index> llrPatch(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> llrWin(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);
  args::ValueFlag<float> λ(parser, "λ", "Threshold", {"lambda"}, 1.f);

  ParseCommand(parser);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());

  Cx5        images = input.readTensor<Cx5>();

  auto hardLLR = [λ = λ.Get()](Cx4 const &xp) {
    Eigen::MatrixXcf patch = CollapseToMatrix(xp);
    auto const       svd = SVD<Cx>(patch.transpose());
    // Soft-threhold svals
    Eigen::VectorXf const s = (svd.S.array().abs() > λ).select(svd.S, 0.f);
    patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
    Cx4 yp = Tensorfy(patch, xp.dimensions());
    return yp;
  };

  Cx4                   img(FirstN<4>(images.dimensions()));
  Cx4                   out(FirstN<4>(images.dimensions()));
  Eigen::TensorMap<Cx4> outmap(out.data(), out.dimensions());
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    img = images.chip<4>(iv);
    Patches(llrPatch.Get(), llrWin.Get(), false, hardLLR, ConstMap(img), outmap);
    images.chip<4>(iv) = out;
  }
  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Image);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
