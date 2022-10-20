#include "pad.hpp"

#include "threads.hpp"

namespace rl {

template <typename Scalar_, int Rank, int ImgRank>
PadOp<Scalar_, Rank, ImgRank>::PadOp(ImgDims const &imgSize, ImgDims const &padSize, OtherDims const &otherSize)
  : Parent(fmt::format(FMT_STRING("{}D PadOp"), ImgRank), Concatenate(otherSize, imgSize), Concatenate(otherSize, padSize))
{
  init();
}

template <typename Scalar_, int Rank, int ImgRank>
PadOp<Scalar_, Rank, ImgRank>::PadOp(ImgDims const &imgSize, OutputMap y)
  : Parent(
      fmt::format(FMT_STRING("{}D PadOp"), ImgRank),
      Concatenate(FirstN<Rank - ImgRank>(y.dimensions()), imgSize),
      y.dimensions())
{
  init();
}

template <typename Scalar_, int Rank, int ImgRank>
PadOp<Scalar_, Rank, ImgRank>::PadOp(InputMap x, OutputMap y)
  : Parent(fmt::format(FMT_STRING("{}D PadOp"), ImgRank), x, y)
{
  if ((Rank > ImgRank) && (FirstN<Rank - ImgRank>(inputDimensions()) != FirstN<Rank - ImgRank>(outputDimensions()))) {
    Log::Fail(FMT_STRING("PadOp input dims {} did not match {}"), inputDimensions(), outputDimensions());
  }
  init();
}

template <typename Scalar_, int Rank, int ImgRank>
auto PadOp<Scalar_, Rank, ImgRank>::forward(InputMap x) const -> OutputMap
{
  auto const time = this->startForward(x);
  this->output().device(Threads::GlobalDevice()) = x.pad(paddings_);
  this->finishForward(this->output(), time);
  return this->output();
}

template <typename Scalar_, int Rank, int ImgRank>
auto PadOp<Scalar_, Rank, ImgRank>::adjoint(OutputMap y) const -> InputMap
{
  auto const time = this->startAdjoint(y);
  this->input().device(Threads::GlobalDevice()) = y.slice(left_, inputDimensions());
  this->finishAdjoint(this->input(), time);
  return this->input();
}

template <typename Scalar_, int Rank, int ImgRank>
void PadOp<Scalar_, Rank, ImgRank>::init()
{
  auto const in = inputDimensions();
  auto const out = outputDimensions();
  for (Index ii = 0; ii < Rank; ii++) {
    if (in[ii] > out[ii]) {
      Log::Fail(FMT_STRING("Padding input dims {} larger than output dims {}"), in, out);
    }
  }
  std::transform(
    out.begin(), out.end(), in.begin(), left_.begin(), [](Index big, Index small) { return (big - small + 1) / 2; });
  std::transform(out.begin(), out.end(), in.begin(), right_.begin(), [](Index big, Index small) { return (big - small) / 2; });
  std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(), [](Index left, Index right) {
    return std::make_pair(left, right);
  });
}

template struct PadOp<Cx, 1, 1>;
template struct PadOp<Cx, 3, 1>;
template struct PadOp<Cx, 2, 2>;
template struct PadOp<Cx, 4, 2>;
template struct PadOp<Cx, 3, 3>;
template struct PadOp<Cx, 4, 3>;
template struct PadOp<Cx, 5, 3>;

} // namespace rl
