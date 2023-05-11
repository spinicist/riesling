#include "types.hpp"

#include "algo/otsu.hpp"
#include "basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_basis_img(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input image file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> nBasis(parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 5);
  args::Flag demean(parser, "C", "Mean-center dynamics", {"demean"});
  args::Flag rotate(parser, "V", "Rotate basis", {"rotate"});
  args::Flag normalize(parser, "N", "Normalize before SVD", {"normalize"});
  ParseCommand(parser, iname);
  if (!oname) {
    throw args::Error("No output filename specified");
  }

  HD5::Reader reader(iname.Get());
  Cx4 const img = reader.readSlab<Cx4>(HD5::Keys::Image, 0);
  Sz4 const shape = img.dimensions();
  Cx4 const ref = img.slice(Sz4{shape[0] - 1, 0, 0, 0}, AddFront(LastN<3>(shape), 1));
  Re4 const real = (img / ref.broadcast(Sz4{shape[0], 1, 1, 1})).real();
  auto const realMat = CollapseToMatrix(real);
  Re3 const toMask = ref.chip<0>(0).abs();
  auto const toMaskMat = CollapseToArray(toMask);
  auto const [thresh, count] = Otsu(toMaskMat);
  Eigen::MatrixXf dynamics(shape[0], count);
  Index col = 0;
  for (Index ii = 0; ii < realMat.cols(); ii++) {
    if (toMaskMat(ii) >= thresh) {
      dynamics.col(col) = realMat.col(ii);
      col++;
    }
  }
  Basis basis(dynamics, 99.f, nBasis.Get(), demean, rotate, normalize);
  HD5::Writer writer(oname.Get());
  basis.write(writer);
  return EXIT_SUCCESS;
}
