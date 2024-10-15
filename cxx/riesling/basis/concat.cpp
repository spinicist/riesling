#include "types.hpp"

#include "basis/basis.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "tensors.hpp"

#include <Eigen/Householder>

using namespace rl;

void main_basis_concat(args::Subparser &parser)
{
  args::Positional<std::string> iname1(parser, "F", "First file");
  args::Positional<std::string> iname2(parser, "F", "Second file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output file");
  args::Flag                    ortho(parser, "O", "Orthogonalize basis", {"ortho"});

  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!iname1) { throw args::Error("Input file 1 not specified"); }
  if (!iname2) { throw args::Error("Input file 2 not specified"); }
  if (!oname) { throw args::Error("Output file not specified"); }

  auto const b1 = LoadBasis(iname1.Get());
  auto const b2 = LoadBasis(iname2.Get());

  if (b1->nSample() != b1->nSample()) { throw Log::Failure(cmd, "Number of samples in both bases must match"); }
  if (b1->nTrace() != b1->nTrace()) { throw Log::Failure(cmd, "Number of traces in both bases must match"); }

  auto const n1 = b1->nB();
  auto const n2 = b2->nB();
  auto const nS = b1->nSample();
  auto const nT = b2->nTrace();

  Cx3 nb(n1 + n2, nS, nT);
  nb.slice(Sz3{0, 0, 0}, Sz3{n1, nS, nT}) = b1->B;
  nb.slice(Sz3{n1, 0, 0}, Sz3{n2, nS, nT}) = b2->B;

  if (ortho) {
    auto const                N = n1 + n2;
    auto const                L = nS * nT;
    auto const                scale = std::sqrt(L);
    Eigen::ArrayXXcf::MapType bmap(nb.data(), N, L);
    auto const                h = bmap.cast<Cxd>().matrix().transpose().householderQr();
    Eigen::MatrixXcd const    I = Eigen::MatrixXcd::Identity(L, N);
    Eigen::MatrixXcd const    Q = h.householderQ() * I;
    Eigen::MatrixXcf          R = h.matrixQR().topRows(N).cast<Cx>().triangularView<Eigen::Upper>();
    R /= scale;
    bmap = Q.transpose().cast<Cx>() * scale;
    Basis b(nb, AsTensorMap(R, Sz2{R.rows(), R.cols()}));
    b.write(oname.Get());
  } else {
    Basis const b3(nb);
    b3.write(oname.Get());
  }
  Log::Print(cmd, "Finished");
}
