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
}

Basis::Basis(Cx3 const &Bb, Re1 const &tt)
  : B{Bb}
  , t{tt}
{
}

Basis::Basis(Cx3 const &Bb, Re1 const &tt, Cx2 const &Rr)
  : B{Bb}
  , R{Rr}
  , t{tt}
{
}

Basis::Basis(Index const nB, Index const nSample, Index const nTrace)
  : B(nB, nSample, nTrace)
{
  B.setConstant(1.f);
}

Basis::Basis(std::string const &fname)
{
  HD5::Reader basisReader(fname);
  B = basisReader.readTensor<Cx3>(HD5::Keys::Basis);
  if (basisReader.exists("time")) {
    t = basisReader.readTensor<Re1>("time");
    if (t.dimension(0) != B.dimension(2)) {
      throw Log::Failure("basis", "{} traces but {} timepoints", B.dimension(2), t.dimension(0));
    }
  } else {
    t.reshape(Sz1{B.dimension(2)});
    for (Index it = 0; it < t.dimension(0); it++) {
      t(it) = it;
    }
  }
  if (basisReader.exists("R")) { R = basisReader.readTensor<Cx2>("R"); }
  if (basisReader.exists("scales")) {
    scales = basisReader.readTensor<Re1>("scales");
    if (scales.dimension(0) != B.dimension(0)) {
      throw Log::Failure("basis", "scales had size {} expected {}", scales.dimension(0), B.dimension(0));
    }
    Log::Print("basis", "Loaded basis scales");
  }
}

auto Basis::nB() const -> Index { return B.dimension(0); }
auto Basis::nSample() const -> Index { return B.dimension(1); }
auto Basis::nTrace() const -> Index { return B.dimension(2); }

auto Basis::entry(Index const s, Index const t) const -> Cx1
{
  if (scales.size()) {
    Cx1 const e = B.chip<2>(t % B.dimension(2)).chip<1>(s % B.dimension(1)) * scales.cast<Cx>();
    return e;
  } else {
    return B.chip<2>(t % B.dimension(2)).chip<1>(s % B.dimension(1));
  }
}

auto Basis::entryConj(Index const s, Index const t) const -> Cx1
{
  if (scales.size()) {
    Cx1 const e = B.chip<2>(t % B.dimension(2)).chip<1>(s % B.dimension(1)).conjugate() * scales.cast<Cx>();
    return e;
  } else {
    return B.chip<2>(t % B.dimension(2)).chip<1>(s % B.dimension(1)).conjugate();
  }
}

void Basis::write(std::string const &basisFile) const
{
  HD5::Writer writer(basisFile);
  writer.writeTensor(HD5::Keys::Basis, B.dimensions(), B.data(), HD5::Dims::Basis);
  if (R.size()) { writer.writeTensor("R", R.dimensions(), R.data(), {"v2", "v1"}); }
  if (t.size()) { writer.writeTensor("time", t.dimensions(), t.data(), {"t"}); }
}

void Basis::concat(Basis const &other)
{
  if (other.nSample() != nSample() || other.nTrace() != nTrace()) {
    throw Log::Failure("Basis", "Incompatible basis dimensions");
  }
  B = Cx3(B.concatenate(other.B, 0));
}

template <int ND> auto Basis::blend(CxN<ND> const &images, Index const is, Index const it) const -> CxN<ND - 1>
{
  if (is < 0 || is >= nSample()) { throw Log::Failure("Basis", "Invalid sample point {}", is); }
  if (it < 0 || it >= nTrace()) { throw Log::Failure("Basis", "Invalid trace point {}", it); }
  Log::Print("basis", "Blending sample {} trace {}", is, it);

  Cx1 b;
  if (scales.size()) {
    b = B.chip<2>(it).chip<1>(is).conjugate() * scales.cast<Cx>();
  } else {
    b = B.chip<2>(it).chip<1>(is).conjugate();
  }
  if (R.size()) {
    return b.contract(R, Eigen::IndexPairList<Eigen::type2indexpair<0, 1>>())
      .contract(images, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
  } else {
    return b.contract(images, Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>());
  }
}

template auto Basis::blend(Cx5 const &, Index const, Index const) const -> Cx4;

template <int ND> void Basis::applyR(CxN<ND> &data) const
{
  if (R.size()) {
    Log::Debug("Basis", "Calculating R inverse");
    Eigen::MatrixXcf::ConstMapType Rm(R.data(), R.dimension(0), R.dimension(1));
    Eigen::MatrixXcf const         Rinv = Rm.inverse();
    auto const                     Rim = AsTensorMap(Rinv, Sz2{Rinv.rows(), Rinv.cols()});
    Log::Print("Basis", "Apply R");
    data.device(Threads::TensorDevice()) = CxN<ND>(Rim.contract(data, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()));
  }
}

template void Basis::applyR(Cx5 &) const;

auto LoadBasis(std::string const &basisFile) -> std::unique_ptr<Basis>
{
  if (basisFile.empty()) {
    return nullptr;
  } else {
    return std::make_unique<Basis>(basisFile);
  }
}

} // namespace rl
