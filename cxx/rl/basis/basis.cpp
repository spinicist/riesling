#include "basis.hpp"

#include "../algo/decomp.hpp"
#include "../algo/stats.hpp"
#include "../io/hd5.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl {

Basis::Basis()
{
  B.resize(1, 1, 1);
  B.setConstant(1.f);
  scale = 1.f;
}

Basis::Basis(Cx3 const &Bb)
  : B{Bb}
{
  scale = std::sqrt(nSample() * nTrace() / nB());
  Log::Print("basis", "nB {} nS {} nT {} scale {}", nB(), nSample(), nTrace(), scale);
}

Basis::Basis(Cx3 const &Bb, Cx2 const &Rr)
  : B{Bb}
  , R{Rr}
{
  scale = std::sqrt(nSample() * nTrace() / nB());
  Log::Print("basis", "nB {} nS {} nT {} scale {}", nB(), nSample(), nTrace(), scale);
}

Basis::Basis(Index const nB_, Index const nS, Index const nT)
  : B(nB_, nS, nT)
{
  scale = 1.f;
  B.setConstant(1.f);
}

auto Basis::nB() const -> Index { return B.dimension(0); }
auto Basis::nSample() const -> Index { return B.dimension(1); }
auto Basis::nTrace() const -> Index { return B.dimension(2); }

auto Basis::entry(Index const s, Index const t) const -> Cx1
{
  return B.chip<2>(t % B.dimension(2)).chip<1>(s % B.dimension(1)) * Cx(scale);
}

auto Basis::entry(Index const b, Index const s, Index const t) const -> Cx
{
  return B(b, s % B.dimension(1), t % B.dimension(2)) * Cx(scale);
}

void Basis::write(std::string const &basisFile) const
{
  HD5::Writer writer(basisFile);
  writer.writeTensor(HD5::Keys::Basis, B.dimensions(), B.data(), HD5::Dims::Basis);
  if (R.size()) { writer.writeTensor("R", R.dimensions(), R.data(), HD5::DNames<2>{"v2", "v1"}); }
}

void Basis::concat(Basis const &other)
{
  if (other.nSample() != nSample() || other.nTrace() != nTrace()) {
    throw Log::Failure("Basis", "Incompatible basis dimensions");
  }
  B = Cx3(B.concatenate(other.B, 0));
}

template <int ND> auto Basis::blend(CxN<ND> const &images, Index const is, Index const it, Index nr) const -> CxN<ND - 1>
{ // Here ND will the order of the full images including the basis and time dimensions
  if (is < 0 || is >= nSample()) { throw Log::Failure("Basis", "Invalid sample point {}", is); }
  if (it < 0 || it >= nTrace()) { throw Log::Failure("Basis", "Invalid trace point {}", it); }
  if (nr > nB()) { throw Log::Failure("Basis", "Requested {} basis vectors but there are only {}", nB(), nr); }
  if (nr < 1) { nr = nB(); }
  Log::Print("Basis", "Blending sample {} trace {} with {} vectors", is, it, nr);
  if (R.size()) {
    return B.chip<2>(it)
             .chip<1>(is)
             .conjugate()
             .contract(R, Eigen::IndexPairList<Eigen::type2indexpair<0, 1>>())
             .slice(Sz1{0}, Sz1{nr})
             .contract(images, Eigen::IndexPairList<Eigen::type2indexpair<0, ND - 2>>()) *
           Cx(scale);
  } else {
    return B.chip<2>(it)
             .chip<1>(is)
             .slice(Sz1{0}, Sz1{nr})
             .conjugate()
             .contract(images, Eigen::IndexPairList<Eigen::type2indexpair<0, ND - 2>>()) *
           Cx(scale);
  }
}

template auto Basis::blend(Cx5 const &, Index const, Index const, Index) const -> Cx4;

template <int ND> void Basis::applyR(CxN<ND> &data) const
{
  if (R.size()) {
    Log::Debug("Basis", "Calculating R inverse");
    Eigen::MatrixXcf::ConstMapType Rm(R.data(), R.dimension(0), R.dimension(1));
    Eigen::MatrixXcf const         Rinv = Rm.inverse();
    auto const                     Rim = AsTensorMap(Rinv, Sz2{Rinv.rows(), Rinv.cols()});
    Log::Print("Basis", "Apply R");
    CxN<ND> temp = Rim.contract(data.shuffle(Sz5{3, 0, 1, 2, 4}), Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>())
                     .shuffle(Sz5{1, 2, 3, 0, 4});
    data.device(Threads::TensorDevice()) = temp;
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
