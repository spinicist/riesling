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

struct MasksToMatricesT
{
  Eigen::MatrixXcf accept, reject;
};

auto MasksToMatrices(Cx4 const &imgs, Re3 const &acceptMask, Re3 const &rejectMask) -> MasksToMatricesT
{
  Index const nSig = Sum(acceptMask);
  Index const nRej = Sum(rejectMask);

  auto const       sz = imgs.dimensions();
  Eigen::MatrixXcf Ω = Eigen::MatrixXcf::Zero(sz[3], nSig);
  Eigen::MatrixXcf Γ = Eigen::MatrixXcf::Zero(sz[3], nRej);
  Index            indexSig = 0, indexRej = 0;
  for (Index iz = 0; iz < sz[2]; iz++) {
    for (Index iy = 0; iy < sz[1]; iy++) {
      for (Index ix = 0; ix < sz[0]; ix++) {
        if (acceptMask(ix, iy, iz)) {
          for (Index ic = 0; ic < sz[3]; ic++) {
            Ω(ic, indexSig) = imgs(ix, iy, iz, ic);
          }
          indexSig++;
        } else if (rejectMask(ix, iy, iz)) {
          for (Index ic = 0; ic < sz[3]; ic++) {
            Γ(ic, indexRej) = imgs(ix, iy, iz, ic);
          }
          indexRej++;
        }
      }
    }
  }
  return {Ω, Γ};
}

void main_rovir(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Index> tp(parser, "T", "Use this time point (default first)", {"tp", 't'}, 0);
  args::ValueFlag<Index> b(parser, "B", "Use this basis coefficient (default first)", {"bc", 'b'}, 0);

  args::ValueFlag<Index> nRetain(parser, "C", "Retain N channels (8, 0 for all)", {"channels", 'n'}, 8);
  args::ValueFlag<float> sirThresh(parser, "S", "SIR threshold (overrides channels)", {"sir"}, 0.f);

  ArrayFlag<float, 3> fovAccept(parser, "F", "ROVir acceptance FoV", {"accept-fov", 'a'}, Eigen::Array<float, 3, 1>::Zero());
  ArrayFlag<float, 3> fovReject(parser, "F", "ROVir rejection FoV", {"reject-fov", 'r'}, Eigen::Array<float, 3, 1>::Zero());

  args::Flag saveImages(parser, "S", "Write ROVir compressed images", {"images"});
  args::Flag debug(parser, "D", "Write debug matrices to .h5", {"debug"});

  ParseCommand(parser, iname, oname);
  auto const cmd = parser.GetCommand().Name();

  HD5::Reader reader(iname.Get());
  auto const  info = reader.readStruct<rl::Info>(HD5::Keys::Info);
  Cx4 const   channelImages = reader.readSlab<Cx4>(HD5::Keys::Data, {HD5::IndexPair{3, b.Get()}, {5, tp.Get()}});
  Re3 const   rss = DimDot<3>(channelImages, channelImages).real().sqrt().log1p(); // For ROI selection
  Sz3 const   sz = rss.dimensions();
  Index const nC = channelImages.dimension(3);
  if (nRetain.Get() > nC) { throw Log::Failure(cmd, "Requested more channels than present in dataset"); }

  Sz3 acceptSt, acceptSz, rejectSt, rejectSz;
  for (Index ii = 0; ii < 3; ii++) {
    Index const voxAccept = 2 * (Index)(fovAccept.Get()[ii] / info.voxel_size[ii] / 2.f);
    Index const voxReject = 2 * (Index)(fovReject.Get()[ii] / info.voxel_size[ii] / 2.f);
    if (voxAccept < 1 || voxAccept > sz[ii]) {
      acceptSt[ii] = 0;
      acceptSz[ii] = sz[ii];
    } else {
      acceptSt[ii] = (sz[ii] - voxAccept) / 2;
      acceptSz[ii] = voxAccept;
    }
    if (voxReject < 1 || voxReject > sz[ii]) {
      rejectSt[ii] = acceptSt[ii];
      rejectSz[ii] = acceptSz[ii];
    } else {
      rejectSt[ii] = (sz[ii] - voxReject) / 2;
      rejectSz[ii] = voxReject;
    }
  }

  auto thresh = Otsu(CollapseToArray(rss)).thresh;
  Log::Print(cmd, "accept threshold {}", thresh);
  Re3 acceptMask(sz), rejectMask(sz);
  acceptMask.setZero();
  acceptMask.slice(acceptSt, acceptSz) = (rss > thresh).slice(acceptSt, acceptSz).cast<float>();
  rejectMask = (rss > thresh).cast<float>();
  rejectMask.slice(rejectSt, rejectSz).setZero();
  auto const [accept, reject] = MasksToMatrices(channelImages, acceptMask, rejectMask);

  /* This needs to be covariance, not correlation. I tested both and correlation doesn't work.
   * Then, Eigen doesn't have a complex generalized eigensolver. Instead, multiply by the inverse
   * of B and then do an SVD, because C will not be self-adjoint
   */
  auto const                                  A = Covariance(accept);
  auto const                                  B = Covariance(reject);
  Eigen::MatrixXcf const                      C = B.inverse() * A;
  Eigen::ComplexEigenSolver<Eigen::MatrixXcf> eig(C);
  Eigen::ArrayXf const                        V = eig.eigenvalues().reverse().cwiseAbs();
  Eigen::MatrixXcf const                      P = eig.eigenvectors().rowwise().reverse();

  // Eigen::ArrayXf const SIR = (P.adjoint() * A * P).array().real() / (P.adjoint() * B * P).array().real();
  Eigen::ArrayXf SIR(nC);
  for (Index ic = 0; ic < nC; ic++) {
     SIR(ic) = (P.col(ic).adjoint() * A * P.col(ic))(0, 0).real() / (P.col(ic).adjoint() * B * P.col(ic))(0, 0).real();
  }
  SIR /= SIR(0);

  auto const             toKeep = sirThresh ? (SIR > sirThresh.Get()).count() : (nRetain.Get() > 0 ? nRetain.Get() : nC);
  Eigen::MatrixXcf const vecs = GramSchmidt(P.leftCols(toKeep), true);
  HD5::Writer            matfile(oname.Get());
  matfile.writeTensor(HD5::Keys::CompressionMatrix, {vecs.rows(), vecs.cols()}, vecs.data(), HD5::DNames<2>{"oc", "ic"});

  if (saveImages) {
    matfile.writeStruct(HD5::Keys::Info, info);
    Cx2CMap   Vm(vecs.data(), {vecs.rows(), vecs.cols()});
    Cx4 const rovirImages = channelImages.contract(Vm, Eigen::IndexPairList<Eigen::type2indexpair<3, 0>>());
    fmt::print(stderr, "dims {}\n", rovirImages.dimensions());
    matfile.writeTensor(HD5::Keys::Data, rovirImages.dimensions(), rovirImages.data(), {"i", "j", "k", "channel"});
  }

  if (debug) {
    matfile.writeTensor("rss", rss.dimensions(), rss.data(), {"i", "j", "k"});
    matfile.writeTensor("accept", acceptMask.dimensions(), acceptMask.data(), {"i", "j", "k"});
    matfile.writeTensor("reject", rejectMask.dimensions(), rejectMask.data(), {"i", "j", "k"});
    matfile.writeTensor("A", {A.rows(), A.cols()}, A.data(), HD5::DNames<2>{"r", "c"});
    matfile.writeTensor("B", {B.rows(), B.cols()}, B.data(), HD5::DNames<2>{"r", "c"});
    matfile.writeTensor("C", {C.rows(), C.cols()}, C.data(), HD5::DNames<2>{"r", "c"});
    auto const eigA = Eig<Cx>(A);
    auto const eigB = Eig<Cx>(B);

    matfile.writeTensor("eigA", {eigA.P.rows(), eigA.P.cols()}, eigA.P.data(), HD5::DNames<2>{"r", "c"});
    matfile.writeTensor("eigB", {eigB.P.rows(), eigB.P.cols()}, eigB.P.data(), HD5::DNames<2>{"r", "c"});
    matfile.writeTensor("eigC", {P.rows(), P.cols()}, P.data(), HD5::DNames<2>{"r", "c"});
  }
  Log::Print(cmd, "Finished");
}
