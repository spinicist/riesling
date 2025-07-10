#include "wavelets.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl::TOps {

template <int ND> auto Wavelets<ND>::PaddedShape(Sz<ND> const shape, std::vector<Index> const dims) -> Sz<ND>
{
  Sz<ND> padded = shape;
  for (auto const d : dims) {
    if (d < 0 || d >= ND) { throw Log::Failure("Waves", "Invalid wavelet dimensions {}", dims); }
    padded[d] = ((shape[d] + 1) / 2) * 2;
  }
  return padded;
}

template <int ND> Wavelets<ND>::Wavelets(Sz<ND> const shape, Index const N, std::vector<Index> const dims)
  : Parent("Waves", shape, shape)
  , N_{N}
  , dims_{dims}
{
  // Check image is adequately padded and bug out if not
  auto const padded = PaddedShape(shape, dims);
  if (shape != padded) { throw Log::Failure(this->name, "Wavelets had dimensions {}, minimum {}", shape, padded); }
  // Daubechie's coeffs courtesy of Wikipedia
  Cc_.resize(N_);
  Cr_.resize(N_);
  switch (N_) {
  case 4: Cc_.setValues({0.6830127f, 1.1830127f, 0.3169873f, -0.1830127f}); break;
  case 6: Cc_.setValues({0.47046721f, 1.14111692f, 0.650365f, -0.19093442f, -0.12083221f, 0.0498175f}); break;
  case 8:
    Cc_.setValues({0.32580343f, 1.01094572f, 0.89220014f, -0.03957503f, -0.26450717f, 0.0436163f, 0.0465036f, -0.01498699f});
    break;
  default: throw(std::runtime_error("Asked for co-efficients that have not been implemented"));
  }
  Cc_ = Cc_ / static_cast<float>(M_SQRT2); // Get scaling correct
  float sign = 1;
  for (Index ii = 0; ii < N_; ii++) {
    Cr_[ii] = sign * Cc_[N_ - 1 - ii];
    sign = -sign;
  }
  Log::Debug("Wave", "Dims {} Coeffs {}", dims_, fmt::streamed(Transpose(Cc_)));
}

template <int ND> void Wavelets<ND>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x * x.constant(s);
  dimLoops(y, false);
  this->finishForward(y, time, false);
}

template <int ND> void Wavelets<ND>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y * y.constant(s);
  dimLoops(x, true);
  this->finishAdjoint(x, time, false);
}

template <int N> auto Range(Index const st = 0, Index const mod = std::numeric_limits<Index>::max()) -> Sz<N>
{
  Sz<N> r;
  for (Index ii = 0; ii < N; ii++) {
    r[ii] = (st + ii) % mod;
  }
  return r;
}

template <int ND> void Wavelets<ND>::dimLoops(InMap x, bool const reverse) const
{
  for (auto const dim : dims_) {
    auto const shuf = Range<ND>(dim, ND);
    auto const otherDims = LastN<ND - 1>(shuf);
    auto const maxSz = ishape[dim];
    auto const otherSz = std::transform_reduce(otherDims.begin(), otherDims.end(), 1L, std::multiplies{},
                                               [ish = this->ishape](size_t const ii) { return ish[ii]; });

    // Work out the smallest wavelet transform we can do on this dimension. Super annoying.
    Index minSz = maxSz;
    while ((minSz / 2) % 2 == 0 && minSz > 4) {
      minSz /= 2;
    }
    auto wav_task = [&](Index const ilo, Index const ihi) {
      for (Index ii = ilo; ii < ihi; ii++) {
        Cx1 temp = x.shuffle(shuf).reshape(Sz2{maxSz, otherSz}).template chip<1>(ii);
        if (reverse) {
          for (Index sz = minSz; sz <= maxSz; sz *= 2) {
            wav1(sz, reverse, temp);
          }
        } else {
          for (Index sz = maxSz; sz >= minSz; sz /= 2) {
            wav1(sz, reverse, temp);
          }
        }
        x.shuffle(shuf).reshape(Sz2{maxSz, otherSz}).template chip<1>(ii) = temp;
      }
    };
    Threads::ChunkFor(wav_task, otherSz);
    Log::Debug("Wave", "Encode dim {}", dim);
  }
}

template <int ND> void Wavelets<ND>::wav1(Index const sz, bool const reverse, Cx1 &x) const
{
  if (sz < 4) return;
  if (sz % 2 == 1) return;

  Cx1 w(sz);
  w.setZero();
  Index const Noff = -N_ / 2;
  Index const hSz = sz / 2;
  if (reverse) {
    for (Index ii = 0; ii < hSz; ii++) {
      Cx const    xLo = x[ii];
      Cx const    xHi = x[ii + hSz];
      Index const index = 2 * ii + Noff;
      for (Index k = 0; k < N_; k++) {
        Index const wrapped = Wrap(index + k, sz);
        w[wrapped] += Cc_[k] * xLo;
        w[wrapped] += Cr_[k] * xHi;
      }
    }
  } else {
    for (Index ii = 0; ii < hSz; ii++) {
      Index const index = 2 * ii + Noff;
      for (Index k = 0; k < N_; k++) {
        Index const wrapped = Wrap(index + k, sz);
        w[ii] += Cc_[k] * x[wrapped];
        w[ii + hSz] += Cr_[k] * x[wrapped];
      }
    }
  }
  x.slice(Sz1{0}, Sz1{sz}) = w;
}

template struct Wavelets<4>;
template struct Wavelets<5>;

} // namespace rl::TOps
