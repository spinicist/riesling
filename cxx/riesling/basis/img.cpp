#include "rl/algo/otsu.hpp"
#include "rl/basis/svd.hpp"
#include "rl/interp.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include "inputs.hpp"

using namespace rl;

void main_basis_img(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input image file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::Flag                        otsu(parser, "O", "Otsu mask", {"otsu"});
  args::ValueFlag<Sz3, SzReader<3>> st(parser, "S", "ROI Start", {"roi-start"});
  args::ValueFlag<Sz3, SzReader<3>> sz(parser, "S", "ROI size", {"roi-size"});
  args::ValueFlag<Index>            spf(parser, "S", "Spokes per frame", {"spf"}, 1);
  args::ValueFlag<Index>            order(parser, "O", "Interpolation order", {"interp-order"}, 3);
  args::Flag                        clamp(parser, "C", "Clamp interpolation", {"interp-clamp"});
  args::ValueFlag<Index>            start2(parser, "S", "Second interp start", {"interp2"});
  args::ValueFlag<Index> nBasis(parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 5);
  args::Flag             demean(parser, "C", "Mean-center dynamics", {"demean"});
  args::Flag             rotate(parser, "V", "Rotate basis", {"rotate"});
  args::Flag             normalize(parser, "N", "Normalize before SVD", {"normalize"});
  ParseCommand(parser, iname);
  auto const cmd = parser.GetCommand().Name();
  if (!oname) { throw args::Error("No output filename specified"); }

  HD5::Reader reader(iname.Get());
  Cx4         img = reader.readSlab<Cx4>(HD5::Keys::Data, {{0, 0}});
  if (st && sz) { img = Cx4(img.slice(AddFront(st.Get(), 0), AddFront(sz.Get(), img.dimension(0)))); }
  Sz4 const  shape = img.dimensions();
  Cx4 const  ref = img.slice(Sz4{shape[0] - 1, 0, 0, 0}, AddFront(LastN<3>(shape), 1));
  Re4 const  real = (img / (ref / ref.abs()).broadcast(Sz4{shape[0], 1, 1, 1})).real();
  auto const realMat = CollapseToConstMatrix(real);
  Re3 const  toMask = ref.chip<0>(0).abs();
  auto const toMaskMat = CollapseToArray(toMask);
  float      thresh;
  Index      count;
  if (otsu) {
    auto o = Otsu(toMaskMat);
    thresh = o.thresh;
    count = o.countAbove;
  } else {
    thresh = 0.f;
    count = Product(LastN<3>(shape));
  }
  Index const     f = shape[0];
  Index const     s = shape[0] * spf.Get();
  Eigen::MatrixXf dynamics(s, count);
  Index           col = 0;
  Eigen::ArrayXi  x1, x2, z1, z2;
  Index           f1 = 0, f2 = 0, s1 = 0, s2 = 0;
  if (start2) {
    f1 = start2.Get();
    f2 = f - f1;
    s1 = f1 * spf.Get();
    s2 = f2 * spf.Get();
    x1 = Eigen::ArrayXi::LinSpaced(f1, 0, s1 - 1) + spf.Get() / 2;
    x2 = Eigen::ArrayXi::LinSpaced(f2, s1, s - 1) + spf.Get() / 2;
    z1 = Eigen::ArrayXi::LinSpaced(s1, 0, s1 - 1);
    z2 = Eigen::ArrayXi::LinSpaced(s2, s1, s - 1);
  } else {
    x1 = Eigen::ArrayXi::LinSpaced(f, 0, s - 1) + spf.Get() / 2;
    z1 = Eigen::ArrayXi::LinSpaced(s, 0, s - 1);
  }
  for (Index ii = 0; ii < realMat.cols(); ii++) {
    if (toMaskMat(ii) >= thresh) {
      if (spf) {
        if (start2) {
          Interpolator interp1(x1, realMat.col(ii).head(f1), order.Get(), clamp);
          Interpolator interp2(x2, realMat.col(ii).tail(f2), order.Get(), clamp);
          dynamics.col(col).head(s1) = interp1(z1);
          dynamics.col(col).tail(s2) = interp2(z2);
        } else {
          Interpolator interp(x1, realMat.col(ii), order.Get(), clamp);
          dynamics.col(col) = interp(z1);
        }
      } else {
        dynamics.col(col) = realMat.col(ii);
      }
      col++;
    }
  }

  SVDBasis const b(dynamics, nBasis.Get(), demean, rotate, normalize);
  HD5::Writer    writer(oname.Get());
  writer.writeTensor(HD5::Keys::Basis, Sz3{b.basis.rows(), 1, b.basis.cols()}, b.basis.data(), HD5::Dims::Basis);
  Log::Print(cmd, "Finished");
}
