#include "wavelets.hpp"
#include "threads.hpp"
#include <fmt/format.h>
#include <iostream>
#include <vector>

namespace rl {

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
{
  // Check image is adequately padded and bug out if not
  auto const padded = PaddedShape(shape, dims);
  if (shape != padded) { Log::Fail("Wavelets had dimensions {}, required {}", shape, padded); }

  for (auto ii = 0; ii < 4; ii++) {
    if (dims[ii]) {
      levels_[ii] = 1;// (Index)std::log2(shape[ii] / (N_ - 1));
    } else {
      levels_[ii] = 0;
    }
  }

  Log::Print("Wavelet levels: {}", levels_);
  // Daubechie's coeffs courtesy of Wikipedia
  D_.resize(N_);
  switch (N_) {
  case 4: D_.setValues({0.6830127f, 1.1830127f, 0.3169873f, -0.1830127f}); break;
  case 6: D_.setValues({0.47046721f, 1.14111692f, 0.650365f, -0.19093442f, -0.12083221f, 0.0498175f}); break;
  case 8:
    D_.setValues({0.32580343f, 1.01094572f, 0.89220014f, -0.03957503f, -0.26450717f, 0.0436163f, 0.0465036f, -0.01498699f});
    break;
  default: Log::Fail("Asked for co-efficients that have not been implemented");
  }
  D_ = D_ / static_cast<float>(M_SQRT2); // Get scaling correct
}

void Wavelets::encode_dim(OutMap &y, Index const dim, Index const level) const
{
  Index const          sz = y.dimension(dim) / (1 << level);
  Index const          hsz = sz / 2;
  Index const          N_2 = N_ / 2;
  Index const          maxDim = 4;
  std::array<Index, 3> otherDims{(dim + 1) % maxDim, (dim + 2) % maxDim, (dim + 3) % maxDim};
  std::sort(otherDims.begin(), otherDims.end(), std::greater{});
  Log::Print("sz {} hsz {} N_2 {} otherDims {}", sz, hsz, N_2, fmt::join(otherDims, ","));
  auto encode_task = [&](Index const ik) {
    Cx1 temp(sz);
    Sz4 ind;
    ind[otherDims[0]] = ik;
    for (Index ij = 0; ij < y.dimension(otherDims[1]); ij++) {
      ind[otherDims[1]] = ij;
      for (Index ii = 0; ii < y.dimension(otherDims[2]); ii++) {
        ind[otherDims[2]] = ii;
        temp.setZero();
        for (Index it = 0; it < hsz; it++) {
          Index f = 1;
          for (Index iw = 0; iw < N_; iw++) {
            ind[dim] = std::clamp(it * 2 + iw, 0L, sz - 1);
            temp(it) += y(ind) * Cx(D_(iw));
            temp(it + hsz) += y(ind) * Cx(D_(N_ - 1 - iw) * f);
            f *= -1;
          }
        }
        for (Index it = 0; it < sz; it++) {
          ind[dim] = it;
          y(ind) = temp(it);
        }
      }
    }
  };
  Log::Print<Log::Level::High>("Wavelets Encode Dimension {} Level {}", dim, level);
  Threads::For(encode_task, y.dimension(otherDims[0]));
}

void Wavelets::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x);
  y = x;
  for (Index dim = 0; dim < 4; dim++) {
    for (Index il = 0; il < levels_[dim]; il++) {
      encode_dim(y, dim, il);
    }
  }
  finishAdjoint(y, time);
}

void Wavelets::decode_dim(InMap &x, Index const dim, Index const level) const
{
  Index const          sz = x.dimension(dim) / (1 << level);
  Index const          hsz = sz / 2;
  Index const          maxDim = 4;
  std::array<Index, 3> otherDims{(dim + 1) % maxDim, (dim + 2) % maxDim, (dim + 3) % maxDim};
  std::sort(otherDims.begin(), otherDims.end(), std::greater{});
  Index const N_2 = N_ / 2;
  auto        decode_task = [&](Index const ik) {
    Cx1 temp(sz);
    Sz4 ind;
    ind[otherDims[0]] = ik;
    for (Index ij = 0; ij < x.dimension(otherDims[1]); ij++) {
      ind[otherDims[1]] = ij;
      for (Index ii = 0; ii < x.dimension(otherDims[2]); ii++) {
        ind[otherDims[2]] = ii;
        temp.setZero();
        for (Index it = 0; it < hsz; it++) {
          Index const temp_index = it * 2;
          for (Index iw = 0; iw < N_2; iw++) {
            Index const line_index = std::clamp(it - iw, 0L, hsz - 1);
            ind[dim] = line_index;
            temp(temp_index) += x(ind) * Cx(D_(iw * 2));
            temp(temp_index + 1) += x(ind) * Cx(D_(iw * 2 + 1));
            ind[dim] = line_index + hsz;
            temp(temp_index) += x(ind) * Cx(D_((N_ - 1) - iw * 2));
            temp(temp_index + 1) -= x(ind) * Cx(D_((N_ - 2) - iw * 2));
          }
        }
        for (Index it = 0; it < sz; it++) {
          ind[dim] = it;
          x(ind) = temp(it);
        }
      }
    }
  };
  Log::Print<Log::Level::High>("Wavelets Decode Dimension {} Level {}", dim, level);
  Threads::For(decode_task, x.dimension(otherDims[0]));
}

void Wavelets::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y);
  x = y;
  for (Index dim = 0; dim < 4; dim++) {
    for (Index il = levels_[dim] - 1; il >= 0; il--) {
      decode_dim(x, dim, il);
    }
  }
  finishAdjoint(x, time);
}

} // namespace rl
