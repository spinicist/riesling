#include "rl/algo/otsu.hpp"
#include "rl/algo/decomp.hpp"
#include "rl/interp.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include "args/all.hpp"

using namespace rl;

void main_basis_img(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input image file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::Flag                        otsu(parser, "O", "Otsu mask", {"otsu"});
  args::ValueFlag<Sz3, SzReader<3>> st(parser, "S", "ROI Start", {"roi-start"});
  args::ValueFlag<Sz3, SzReader<3>> sz(parser, "S", "ROI size", {"roi-size"});
  args::ValueFlag<Index>            tpf(parser, "T", "Traces per frame (expand at end)", {"tpf"}, 1);
  args::ValueFlag<Index> nRetain(parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 8);
  args::Flag             demean(parser, "C", "Mean-center dynamics", {"demean"});
  args::Flag             normalize(parser, "N", "Normalize before SVD", {"normalize"});
  args::Flag             equalize(parser, "E", "Equalize variance in basis", {"equalize"});
  
  ParseCommand(parser, iname);
  auto const cmd = parser.GetCommand().Name();
  if (!oname) { throw args::Error("No output filename specified"); }

  HD5::Reader reader(iname.Get());
  Cx5         img = reader.readTensor<Cx5>();
  if (st && sz) { img = Cx5(img.slice(Concatenate(st.Get(), LastN<2>(img.dimensions())),
   																	  Concatenate(sz.Get(), LastN<2>(img.dimensions())))); }
  Sz5 const  shape = img.dimensions();
  Index const nB = img.dimension(3);
  Index const nT = img.dimension(4);
  Cx3 ref = img.chip<4>(nT - 1).chip<3>(nB - 1);
  Re3 aref = ref.abs();
  Sz5 const arsh = AddBack(ref.dimensions(), 1, 1);
  Sz5 const abrd = AddBack(Constant<3>(1), nB, nT);
  Re5 const  real = (img / (ref / aref).reshape(arsh).broadcast(abrd)).real();
  auto const realMat = CollapseToConstMatrix<Re5, 3>(real);
  auto toMaskMat = CollapseToArray(aref);
  auto const o = otsu ? Otsu(toMaskMat) : OtsuReturn{0.f, Product(FirstN<3>(shape))};
  Eigen::MatrixXcf dynamics(o.countAbove, nB * nT);
  Index ir = 0;
  for (Index ii = 0; ii < realMat.rows(); ii++) {
    if (toMaskMat(ii) >= o.thresh) {
      dynamics.row(ir) = realMat.row(ii).cast<Cx>();
      toMaskMat(ii) = 1.f;
      ir++;
    } else {
    	toMaskMat(ii) = 0.f;
    }
  }
  if (ir != o.countAbove) { throw Log::Failure(cmd, "Programmer error"); }
  if (normalize) { dynamics.rowwise().normalize(); }
  Log::Print(cmd, "Computing SVD {}x{}", dynamics.rows(), dynamics.cols());
  SVD<Cx> svd(dynamics);
  Log::Print(cmd, "Variance: {}\n", svd.variance(nRetain.Get()));
  Eigen::MatrixXcf const basis = equalize ? svd.equalized(nRetain.Get()) : svd.basis(nRetain.Get(), false);
  Log::Print(cmd, "Computing projection");
  Eigen::MatrixXcf const projection = basis.conjugate() * dynamics.transpose();
  Eigen::MatrixXcf const reproj = (basis.transpose() * projection).transpose();
  float const resid = (reproj - dynamics).norm() / dynamics.norm();
  Log::Print(cmd, "Residual {}%", 100 * resid);

  HD5::Writer    writer(oname.Get());
  if (tpf) {
  	Eigen::MatrixXcf const expanded = basis.replicate(tpf.Get(), 1).reshaped(basis.rows(), basis.cols() * tpf.Get());
  	writer.writeTensor(HD5::Keys::Basis, Sz3{expanded.rows(), 1, expanded.cols()}, expanded.data(), HD5::Dims::Basis);
  } else {
  	writer.writeTensor(HD5::Keys::Basis, Sz3{basis.rows(), 1, basis.cols()}, basis.data(), HD5::Dims::Basis);
  }
  writer.writeTensor("mask", aref.dimensions(), aref.data(), {"i", "j", "k"});
  writer.writeTensor(HD5::Keys::Dynamics, Sz2{dynamics.rows(), dynamics.cols()}, dynamics.data(), {"d", "t"});
  writer.writeTensor("projection", Sz2{reproj.rows(), reproj.cols()}, reproj.data(), {"d", "t"});
  Log::Print(cmd, "Finished");
}
