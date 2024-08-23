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

auto Basis::nB() const -> Index { return B.dimension(0); }
auto Basis::nSample() const -> Index { return B.dimension(1); }
auto Basis::nTrace() const -> Index { return B.dimension(2); }

auto Basis::entry(Index const s, Index const t) const -> Cx1 {
  return B.chip<2>(t % B.dimension(2)).chip<1>(s % B.dimension(1));
}

auto Basis::entryConj(Index const s, Index const t) const -> Cx1 {
  return B.chip<2>(t % B.dimension(2)).chip<1>(s % B.dimension(1)).conjugate();
}

void Basis::write(std::string const &basisFile) const
{
  HD5::Writer writer(basisFile);
  writer.writeTensor(HD5::Keys::Basis, B.dimensions(), B.data(), HD5::Dims::Basis);
  if (R.size()) { writer.writeTensor("R", R.dimensions(), R.data(), {"v2", "v1"}); }
}

void Basis::concat(Basis const &other)
{
  if (other.nSample() != nSample() || other.nTrace() != nTrace()) { Log::Fail("Incompatible basis dimensions"); }
  B = Cx3(B.concatenate(other.B, 0));
}

template <int ND> auto Basis::blend(CxN<ND> const &images, Index const is, Index const it) const -> CxN<ND - 1>
{
  if (is < 0 || is >= nSample()) { Log::Fail("Invalid sample point {}", is); }
  if (it < 0 || it >= nTrace()) { Log::Fail("Invalid trace point {}", it); }
  if (R.size()) {
    return B.chip<2>(it)
      .chip<1>(is)
      .conjugate()
      .contract(R, Eigen::IndexPairList<Eigen::type2indexpair<0, 1>>())
      .contract(images, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
  } else {
    Log::Print("No R matrix");
    return B.chip<2>(it).chip<1>(is).conjugate().contract(images, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
  }
}

template auto Basis::blend(Cx5 const &, Index const, Index const) const -> Cx4;

template <int ND> void Basis::applyR(CxN<ND> &data) const
{
  if (R.size()) {
    Log::Debug("Calculating R inverse");
    Eigen::MatrixXcf::ConstMapType Rm(R.data(), R.dimension(0), R.dimension(1));
    Eigen::MatrixXcf const         Rinv = Rm.inverse();
    auto const                     Rim = Tensorfy(Rinv, Sz2{Rinv.rows(), Rinv.cols()});
    Log::Print("Apply R");
    data.device(Threads::GlobalDevice()) = CxN<ND>(Rim.contract(data, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()));
  }
}

template void Basis::applyR(Cx5 &) const;

auto LoadBasis(std::string const &basisFile) -> std::unique_ptr<Basis>
{
  if (basisFile.empty()) {
    return nullptr;
  } else {
    HD5::Reader basisReader(basisFile);
    Cx3 const   B = basisReader.readTensor<Cx3>(HD5::Keys::Basis);
    if (basisReader.exists("R")) {
      Cx2 const R = basisReader.readTensor<Cx2>("R");
      return std::make_unique<Basis>(B, R);
    } else {
      return std::make_unique<Basis>(B);
    }
  }
}

} // namespace rl
