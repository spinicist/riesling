#include "wavelets.hpp"
#include "tensors.hpp"
#include "threads.hpp"
#include <fmt/format.h>
#include <iostream>
#include <vector>

namespace rl::TOps {

auto Wavelets::PaddedShape(Sz4 const shape, Sz4 const dims) -> Sz4
{
  Sz4 padded;
  for (Index ii = 0; ii < 4; ii++) {
    if (dims[ii]) {
      padded[ii] = ((shape[ii] + 1) / 2) * 2;
    } else {
      padded[ii] = shape[ii];
    }
  }
  return padded;
}

Wavelets::Wavelets(Sz4 const shape, Index const N, Sz4 const dims)
  : Parent("WaveletsOp", shape, shape)
  , N_{N}
  , encodeDims_{dims}
{
  // Check image is adequately padded and bug out if not
  auto const padded = PaddedShape(shape, dims);
  if (shape != padded) { Log::Fail("Wavelets had dimensions {}, required {}", shape, padded); }
  // Daubechie's coeffs courtesy of Wikipedia
  Cc_.resize(N_);
  Cr_.resize(N_);
  switch (N_) {
  case 4: Cc_.setValues({0.6830127f, 1.1830127f, 0.3169873f, -0.1830127f}); break;
  case 6: Cc_.setValues({0.47046721f, 1.14111692f, 0.650365f, -0.19093442f, -0.12083221f, 0.0498175f}); break;
  case 8:
    Cc_.setValues({0.32580343f, 1.01094572f, 0.89220014f, -0.03957503f, -0.26450717f, 0.0436163f, 0.0465036f, -0.01498699f});
    break;
  default: Log::Fail("Asked for co-efficients that have not been implemented");
  }
  Cc_ = Cc_ / static_cast<float>(M_SQRT2); // Get scaling correct
  float sign = 1;
  for (Index ii = 0; ii < N_; ii++) {
    Cr_[ii] = sign * Cc_[N_ - 1 - ii];
    sign = -sign;
  }
  Log::Print("Wavelet dimensions: {}", encodeDims_);
  Log::Print("Coeffs: {}", fmt::streamed(Transpose(Cc_)));
}

void Wavelets::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x);
  y = x;
  dimLoops(y, false);
  finishForward(y, time);
}

void Wavelets::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y);
  x = y;
  dimLoops(x, true);
  finishAdjoint(x, time);
}

void Wavelets::dimLoops(InMap &x, bool const reverse) const
{
  Index const maxDim = 4;
  for (Index dim = 0; dim < 4; dim++) {
    if (encodeDims_[dim] != 0) {
      std::array<Index, 3> otherDims{(dim + 1) % maxDim, (dim + 2) % maxDim, (dim + 3) % maxDim};
      std::sort(otherDims.begin(), otherDims.end(), std::less{});
      // Work out the smallest wavelet transform we can do on this dimension. Super annoying.
      Index const maxSz = x.dimension(dim);
      Index       minSz = maxSz;
      while ((minSz / 2) % 2 == 0 && minSz > 4) {
        minSz /= 2;
      }
      auto wav_task = [&](Index const ik) {
        for (Index ij = 0; ij < x.dimension(otherDims[1]); ij++) {
          for (Index ii = 0; ii < x.dimension(otherDims[0]); ii++) {
            Cx1 temp = x.chip(ik, otherDims[2]).chip(ij, otherDims[1]).chip(ii, otherDims[0]);
            if (reverse) {
              for (Index sz = minSz; sz <= maxSz; sz *= 2) {
                wav1(sz, reverse, temp);
              }
            } else {
              for (Index sz = maxSz; sz >= minSz; sz /= 2) {
                wav1(sz, reverse, temp);
              }
            }
            x.chip(ik, otherDims[2]).chip(ij, otherDims[1]).chip(ii, otherDims[0]) = temp;
          }
        }
      };
      Threads::For(wav_task, x.dimension(otherDims[2]));
      Log::Debug("Wavelets Encode Dimension {}", dim);
    }
  }
}

void Wavelets::wav1(Index const sz, bool const reverse, Cx1 &x) const
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
        // fmt::print(stderr, "index {} wrapped {}\n", index, wrapped);
        w[ii] += Cc_[k] * x[wrapped];
        w[ii + hSz] += Cr_[k] * x[wrapped];
      }
    }
  }
  x.slice(Sz1{0}, Sz1{sz}) = w;
}

} // namespace rl::TOps
