#include "pad.hpp"

#include "log.hpp"
#include "threads.hpp"

namespace rl::TOps {

template <typename Scalar_, int Rank, int ImgRank>
Pad<Scalar_, Rank, ImgRank>::Pad(ImgDims const &imgSize, ImgDims const &padSize, OtherDims const &otherSize)
  : Parent(fmt::format("{}D Pad", ImgRank), Concatenate(otherSize, imgSize), Concatenate(otherSize, padSize))
{
  init();
}

template <typename Scalar_, int Rank, int ImgRank>
Pad<Scalar_, Rank, ImgRank>::Pad(ImgDims const &imgSize, OutDims const os)
  : Parent(fmt::format("{}D Pad", ImgRank), Concatenate(FirstN<Rank - ImgRank>(os), imgSize), os)
{
  init();
}

template <typename Scalar_, int Rank, int ImgRank> void Pad<Scalar_, Rank, ImgRank>::init()
{
  for (Index ii = 0; ii < Rank; ii++) {
    if (ishape[ii] > oshape[ii]) { Log::Fail("Padding input dims {} larger than output dims {}", ishape, oshape); }
  }
  std::transform(oshape.begin(), oshape.end(), ishape.begin(), left_.begin(),
                 [](Index big, Index small) { return (big - small + 1) / 2; });
  std::transform(oshape.begin(), oshape.end(), ishape.begin(), right_.begin(),
                 [](Index big, Index small) { return (big - small) / 2; });
  std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <typename Scalar_, int Rank, int ImgRank> void Pad<Scalar_, Rank, ImgRank>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y);
  y.device(Threads::GlobalDevice()) = x.pad(paddings_);
  this->finishForward(y, time);
}

template <typename Scalar_, int Rank, int ImgRank> void Pad<Scalar_, Rank, ImgRank>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x);
  x.device(Threads::GlobalDevice()) = y.slice(left_, ishape);
  this->finishAdjoint(x, time);
}

template <typename Scalar_, int Rank, int ImgRank> void Pad<Scalar_, Rank, ImgRank>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y);
  y.device(Threads::GlobalDevice()) += x.pad(paddings_);
  this->finishForward(y, time);
}

template <typename Scalar_, int Rank, int ImgRank> void Pad<Scalar_, Rank, ImgRank>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x);
  x.device(Threads::GlobalDevice()) += y.slice(left_, ishape);
  this->finishAdjoint(x, time);
}

template struct Pad<float, 1, 1>;
template struct Pad<float, 2, 2>;
template struct Pad<float, 3, 3>;

template struct Pad<Cx, 1, 1>;
template struct Pad<Cx, 3, 1>;
template struct Pad<Cx, 2, 2>;
template struct Pad<Cx, 4, 1>;
template struct Pad<Cx, 4, 2>;
template struct Pad<Cx, 3, 3>;
template struct Pad<Cx, 4, 3>;
template struct Pad<Cx, 5, 2>;
template struct Pad<Cx, 5, 3>;
template struct Pad<Cx, 6, 3>;

} // namespace rl::TOps
