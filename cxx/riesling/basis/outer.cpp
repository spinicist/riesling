#include "rl/basis/basis.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include "inputs.hpp"

#include <Eigen/Householder>

using namespace rl;

void main_basis_outer(args::Subparser &parser)
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

  auto b1 = LoadBasis(iname1.Get());
  auto b2 = LoadBasis(iname2.Get());

  if (b1->nTrace() != 1) { throw Log::Failure(cmd, "Expected 1st basis to have 1 trace"); }
  if (b2->nSample() != 1) { throw Log::Failure(cmd, "Expected 2nd basis to have 1 sample"); }

  auto const n1 = b1->nB();
  auto const n2 = b2->nB();
  auto const nS = b1->nSample();
  auto const nT = b2->nTrace();
  Cx3        nb(n1 * n2, nS, nT);

  for (Index i2 = 0; i2 < n2; i2++) {
    for (Index i1 = 0; i1 < n1; i1++) {
      for (Index is = 0; is < nS; is++) {
        for (Index it = 0; it < nT; it++) {
          nb(i2 * n1 + i1, is, it) = b1->B(i1, is, 0) * b2->B(i2, 0, it);
        }
      }
    }
  }

  if (ortho) {
    auto const                M = nS * nT;
    auto const                N = n1 * n2;
    Eigen::ArrayXXcf::MapType bmap(nb.data(), N, M);
    auto const                h = bmap.cast<Cxd>().matrix().transpose().householderQr();
    Eigen::MatrixXcd const    I = Eigen::MatrixXcd::Identity(M, N);
    Eigen::MatrixXcd const    Q = h.householderQ() * I;
    Eigen::MatrixXcf const    R = h.matrixQR().topRows(N).cast<Cx>().triangularView<Eigen::Upper>();
    bmap = Q.transpose().cast<Cx>() * std::sqrt(M);
    Basis b(nb, b1->t, AsTensorMap(R, Sz2{R.rows(), R.cols()}));
    b.write(oname.Get());
  } else {
    Basis const b3(nb, b1->t);
    b3.write(oname.Get());
  }
  Log::Print(cmd, "Finished");
}
