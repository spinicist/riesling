#include "rl/algo/decomp.hpp"
#include "rl/io/writer.hpp"
#include "rl/log/log.hpp"
#include "rl/types.hpp"

#include "inputs.hpp"

#include <complex>
#include <numbers>

using namespace std::literals::complex_literals;

void main_basis_decay(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> nRetain(parser, "N", "Number of basis vectors", {"N", 'N'}, 4);
  args::ValueFlag<Index> Ne(parser, "E", "Number of echoes", {"Necho", 'e'}, 8);
  args::ValueFlag<float> esp(parser, "E", "Echo spacing", {"esp"}, 10.e-3f);
  args::ValueFlag<float> t2min(parser, "T", "Minimum T2", {"t2min"}, 10.e-3f);
  args::ValueFlag<float> t2max(parser, "T", "Maximum T2", {"t2max"}, 500.e-3f);
  args::ValueFlag<Index> Nt2(parser, "N", "Number of T2s", {"nt2"}, 16);
  args::Flag             centric(parser, "C", "Centre-out view order", {"centric"});

  ParseCommand(parser, oname);
  auto const cmd = parser.GetCommand().Name();
  if (!oname) { throw args::Error("No output filename specified"); }
  Eigen::MatrixXcf dynamics(Nt2.Get(), Ne.Get());
  for (Index it = 0; it < Nt2.Get(); it++) {
    float const t2 = t2min.Get() + it * (t2max.Get() - t2min.Get()) / (Nt2.Get() - 1.f);
    for (Index ie = 0; ie < Ne.Get(); ie++) {
      float const t = esp.Get() * ie;
      dynamics(it, ie) = exp(-t / t2);
    }
  }

  rl::Log::Print(cmd, "Computing SVD {}x{}", dynamics.rows(), dynamics.cols());
  rl::SVD<rl::Cx>  svd(dynamics);
  Eigen::MatrixXcf basis = svd.basis(nRetain.Get());
  rl::Log::Print(cmd, "Computing projection");
  fmt::print(stderr, "dynamics {} {} basis {} {}\n", dynamics.rows(), dynamics.cols(), basis.rows(), basis.cols());
  Eigen::MatrixXcf temp = dynamics * basis.adjoint();
  fmt::print(stderr, "temp {} {}\n", temp.rows(), temp.cols());
  Eigen::MatrixXcf proj = temp * basis;
  auto const       resid = (dynamics - proj).norm() / dynamics.norm();
  rl::Log::Print(cmd, "Residual {}%", 100 * resid);

  if (centric) {
    Eigen::MatrixXcf const original = basis;
    Index const            h = Ne.Get() / 2;
    basis.setZero();
    for (Index ii = 0; ii < h; ii++) {
      fmt::print(stderr, "ii {} ii*2+1 {} p {} m {}\n", ii, ii * 2 + 1, h + ii, h - 1 - ii);
      basis.col(h + ii) = original.col(ii * 2);
      basis.col(h - 1 - ii) = original.col(ii * 2 + 1);
    }
  }
  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, rl::Sz3{basis.rows(), basis.cols(), 1}, basis.data(), rl::HD5::Dims::Basis);
  writer.writeTensor(rl::HD5::Keys::Dynamics, rl::Sz2{dynamics.rows(), dynamics.cols()}, dynamics.data(), {"T2", "T"});
  writer.writeTensor("projection", rl::Sz2{proj.rows(), proj.cols()}, proj.data(), {"T2", "T"});

  rl::Log::Print(cmd, "Finished");
}
