#include "basis.hpp"

#include "algo/decomp.hpp"
#include "algo/stats.hpp"
#include "io/hd5.hpp"
#include "tensors.hpp"
#include "threads.hpp"

namespace rl {

Basis::Basis()
{
  B.resize(1, 1, 1);
  B.setConstant(1.f);
}

Basis::Basis(Cx3 const &Bb)
  : B{Bb}
{
}

Basis::Basis(Cx3 const &Bb, Cx2 const &Rr)
  : B{Bb}
  , R{Rr}
{
}

Basis::Basis(Index const nB, Index const nSample, Index const nTrace)
  : B(nB, nSample, nTrace)
{
  B.setConstant(1.f);
}

Basis::Basis(std::string const &basisFile)
{
  if (basisFile.empty()) {
    B.resize(1, 1, 1);
    B.setConstant(1.f);
  } else {
    HD5::Reader basisReader(basisFile);
    B = basisReader.readTensor<Cx3>(HD5::Keys::Basis);
    if (basisReader.exists("R")) {
      R = basisReader.readTensor<Cx2>("R");
    }
  }
}

void Basis::write(std::string const &basisFile) const
{
  HD5::Writer writer(basisFile);
  writer.writeTensor(HD5::Keys::Basis, B.dimensions(), B.data(), HD5::Dims::Basis);
  if (R.size()) {
    writer.writeTensor("R", R.dimensions(), R.data(), {"v2", "v1"});
  }
}

auto Basis::nV() const -> Index { return B.dimension(0); }
auto Basis::nSample() const -> Index { return B.dimension(1); }
auto Basis::nTrace() const -> Index { return B.dimension(2); }

template <int ND> void Basis::applyR(CxN<ND> &data) const
{
  if (R.size()) {
    Log::Debug("Calculating R inverse");
    Eigen::MatrixXcf::ConstMapType Rm(R.data(), R.dimension(0), R.dimension(1));
    Eigen::MatrixXcf const Rinv = Rm.inverse();
    auto const             Rim = Tensorfy(Rinv, Sz2{Rinv.rows(), Rinv.cols()});
    Log::Print("Apply R");
    data.device(Threads::GlobalDevice()) = CxN<ND>(Rim.contract(data, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()));
  }
}

template void Basis::applyR(Cx5 &) const;

} // namespace rl
