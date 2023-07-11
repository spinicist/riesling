#include "pad.hpp"

#include "log.hpp"
#include "threads.hpp"

namespace rl {

template <typename Scalar_, int Rank, int ImgRank>
PadOp<Scalar_, Rank, ImgRank>::PadOp(ImgDims const &imgSize, ImgDims const &padSize, OtherDims const &otherSize)
  : Parent(fmt::format("{}D PadOp", ImgRank), Concatenate(otherSize, imgSize), Concatenate(otherSize, padSize))
{
  init();
}

template <typename Scalar_, int Rank, int ImgRank>
PadOp<Scalar_, Rank, ImgRank>::PadOp(ImgDims const &imgSize, OutDims os)
  : Parent(fmt::format("{}D PadOp", ImgRank), Concatenate(FirstN<Rank - ImgRank>(os), imgSize), os)
{
  init();
}

template <typename Scalar_, int Rank, int ImgRank>
void PadOp<Scalar_, Rank, ImgRank>::init()
{
  for (Index ii = 0; ii < Rank; ii++) {
    if (ishape[ii] > oshape[ii]) { Log::Fail("Padding input dims {} larger than output dims {}", ishape, oshape); }
  }
  std::transform(
    oshape.begin(), oshape.end(), ishape.begin(), left_.begin(), [](Index big, Index small) { return (big - small + 1) / 2; });
  std::transform(
    oshape.begin(), oshape.end(), ishape.begin(), right_.begin(), [](Index big, Index small) { return (big - small) / 2; });
  std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(), [](Index left, Index right) {
    return std::make_pair(left, right);
  });
}

template <typename Scalar_, int Rank, int ImgRank>
void PadOp<Scalar_, Rank, ImgRank>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  y.device(Threads::GlobalDevice()) = x.pad(paddings_);
  this->finishForward(y, time);
}

template <typename Scalar_, int Rank, int ImgRank>
void PadOp<Scalar_, Rank, ImgRank>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  x.device(Threads::GlobalDevice()) = y.slice(left_, ishape);
  this->finishAdjoint(x, time);
}

template struct PadOp<Cx, 1, 1>;
template struct PadOp<Cx, 3, 1>;
template struct PadOp<Cx, 2, 2>;
template struct PadOp<Cx, 4, 2>;
template struct PadOp<Cx, 3, 3>;
template struct PadOp<Cx, 4, 3>;
template struct PadOp<Cx, 5, 3>;

} // namespace rl
