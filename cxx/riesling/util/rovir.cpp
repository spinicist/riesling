#include "inputs.hpp"
#include "rl/algo/decomp.hpp"
#include "rl/algo/gs.hpp"
#include "rl/algo/otsu.hpp"
#include "rl/algo/stats.hpp"
#include "rl/basis/basis.hpp"
#include "rl/compressor.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include <Eigen/Eigenvalues>

using namespace rl;

void main_rovir(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Index> tp(parser, "T", "Use this time point (default first)", {"tp", 't'}, 0);
  args::ValueFlag<Index> b(parser, "B", "Use this basis coefficient (default first)", {"bc", 'b'}, 0);

  args::ValueFlag<Index> nRetain(parser, "C", "Retain N channels (8, 0 for all)", {"channels", 'n'}, 8);
  args::ValueFlag<float> energy(parser, "E", "Retain fraction energy (overrides channels)", {"energy"}, -1.f);

  args::ValueFlag<float> fov(parser, "F", "ROVIR Signal FoV", {"rovir-fov"}, 1024.f);
  args::ValueFlag<Index> gap(parser, "G", "ROVIR gap in voxels", {"rovir-gap"}, 0);

  ParseCommand(parser, iname, oname);
  auto const cmd = parser.GetCommand().Name();

  HD5::Reader reader(iname.Get());
  HD5::Writer matfile(oname.Get());
  Cx4 const   channelImages = reader.readSlab<Cx4>(HD5::Keys::Data, {HD5::IndexPair{3, b.Get()}, {5, tp.Get()}});
  Re3 const   rss = DimDot<3>(channelImages, channelImages).real().sqrt().log1p(); // For ROI selection
  Sz3 const   sz = rss.dimensions();
  Index const nC = channelImages.dimension(3);
  if (nRetain.Get() > nC) { throw Log::Failure(cmd, "Requested more channels than present in dataset"); }
  auto thresh = Otsu(CollapseToArray(rss)).thresh;
  Log::Print(cmd, "Signal threshold {}", thresh);
  Re3 signalMask(sz), rejectMask(sz);
  signalMask.setZero();
  rejectMask = (rss > thresh).cast<float>();
  Sz3 const st{0, 0, sz[2] / 2};
  Sz3 const sl{sz[0], sz[1], sz[2] / 2};
  signalMask.slice(st, sl) = rejectMask.slice(st, sl);
  rejectMask.slice(st, sl).setZero();
  matfile.writeTensor("rss", rss.dimensions(), rss.data(), {"i", "j", "k"});
  matfile.writeTensor("signal", signalMask.dimensions(), signalMask.data(), {"i", "j", "k"});
  matfile.writeTensor("reject", rejectMask.dimensions(), rejectMask.data(), {"i", "j", "k"});
  // Copy to A & B matrices
  Index const nSig = Sum(signalMask);
  Index const nRej = Sum(rejectMask);
  Log::Print(cmd, "{} voxels in signal mask, {} in interference mask", nSig, nRej);

  Eigen::MatrixXcf Ω = Eigen::MatrixXcf::Zero(nC, nSig);
  Eigen::MatrixXcf Γ = Eigen::MatrixXcf::Zero(nC, nRej);
  Index            indexSig = 0, indexRej = 0;
  for (Index iz = 0; iz < sz[2]; iz++) {
    for (Index iy = 0; iy < sz[1]; iy++) {
      for (Index ix = 0; ix < sz[0]; ix++) {
        if (signalMask(ix, iy, iz)) {
          for (Index ic = 0; ic < nC; ic++) {
            Ω(ic, indexSig) = channelImages(ix, iy, iz, ic);
          }
          indexSig++;
        } else if (rejectMask(ix, iy, iz)) {
          for (Index ic = 0; ic < nC; ic++) {
            Γ(ic, indexRej) = channelImages(ix, iy, iz, ic);
          }
          indexRej++;
        }
      }
    }
  }
  
  auto const A = Correlation(Ω);
  auto const B = Correlation(Γ);
  auto const eigA = Eig<Cx>(A);
  auto const eigB = Eig<Cx>(B);
  Eigen::MatrixXcf const C = B.inverse() * A;
  auto const eigC = Eig<Cx>(C);

  matfile.writeTensor("A", {A.rows(), A.cols()}, A.data(), HD5::DNames<2>{"r", "c"});
  matfile.writeTensor("B", {B.rows(), B.cols()}, B.data(), HD5::DNames<2>{"r", "c"});
  matfile.writeTensor("C", {C.rows(), C.cols()}, C.data(), HD5::DNames<2>{"r", "c"});
  matfile.writeTensor("eigA", {eigA.P.rows(), eigA.P.cols()}, eigA.P.data(), HD5::DNames<2>{"r", "c"});
  matfile.writeTensor("eigB", {eigB.P.rows(), eigB.P.cols()}, eigB.P.data(), HD5::DNames<2>{"r", "c"});
  matfile.writeTensor("eigC", {eigC.P.rows(), eigC.P.cols()}, eigC.P.data(), HD5::DNames<2>{"r", "c"});

  // Eigen::LLT<Eigen::MatrixXcf> const cholB(B);
  // Eigen::MatrixXcf                   C2 = A.selfadjointView<Eigen::Lower>();
  // cholB.matrixL().solveInPlace<Eigen::OnTheLeft>(C2);
  // cholB.matrixU().solveInPlace<Eigen::OnTheRight>(C2);
  // auto const eigC2 = Eig<Cx>(C2);
  // matfile.writeTensor("C2", {C2.rows(), C2.cols()}, C2.data(), HD5::DNames<2>{"r", "c"});
  // matfile.writeTensor("eigC2", {eigC2.P.rows(), eigC2.P.cols()}, eigC2.P.data(), HD5::DNames<2>{"r", "c"});

  Index const toKeep = nRetain.Get() > 0 ? nRetain.Get() : nC;
  auto const n = energy ? CountBelow(eigC.V, energy.Get()) : toKeep;
  Eigen::MatrixXcf vecs = eigC.P.leftCols(n);
  // cholB.matrixU().solveInPlace(vecs);
  // auto const psi = GramSchmidt(vecs);
  matfile.writeTensor(HD5::Keys::CompressionMatrix, {vecs.rows(), vecs.cols()}, vecs.data(), HD5::DNames<2>{"oc", "ic"});
}
