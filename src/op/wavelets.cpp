#include "wavelets.hpp"
#include "threads.hpp"
#include <fmt/format.h>
#include <iostream>
#include <vector>

// Once the first chip is done the Tensor will be 2D and oh look I've gone cross-eyed
namespace {
std::vector<Index> const dim1 = {1, 2, 0};
std::vector<Index> const dim2 = {1, 0, 0};
} // namespace

namespace rl {

auto Wavelets::PaddedDimensions(Sz4 const dims, Index const L) -> Sz4
{
  Sz4 padded;
  padded[0] = dims[0];
  for (Index ii = 1; ii < 4; ii++) {
    Index const d = dims[ii];
    Index const sz = 1 << L;
    auto const res = std::div(d, sz);
    if (res.rem) {
      padded[ii] = (res.quot + 1) * sz;
    } else {
      padded[ii] = res.quot * sz;
    }
  }
  return padded;
}

Wavelets::Wavelets(Sz4 const dims, Index const N, Index const L)
  : dims_{dims}
  , N_{N}
  , L_{L}
  , ws_{dims_}
{
  // Check image is adequately padded and bug out if not
  auto const padded = PaddedDimensions(dims_, L_);
  if (dims_ != padded) {
    Log::Fail(FMT_STRING("Wavelets with {} levels had dimensions {}, required {}"), L_, dims_, padded);
  }

  // Daubechie's coeffs courtesy of Wikipedia
  D_.resize(N);
  switch (N) {
  case 4:
    D_.setValues({0.6830127f, 1.1830127f, 0.3169873f, -0.1830127f});
    break;
  case 6:
    D_.setValues({0.47046721f, 1.14111692f, 0.650365f, -0.19093442f, -0.12083221f, 0.0498175f});
    break;
  case 8:
    D_.setValues(
      {0.32580343f, 1.01094572f, 0.89220014f, -0.03957503f, -0.26450717f, 0.0436163f, 0.0465036f, -0.01498699f});
    break;
  default:
    Log::Fail("Asked for co-efficients that have not been implemented");
  }
  D_ = D_ / static_cast<float>(M_SQRT2); // Get scaling correct
  Log::Print(FMT_STRING("Wavelets: Dims {} N={} L={}"), dims_, N_, L_);
}

// std::tuple<Sz3, Sz3> Wavelets::pad_setup(Sz3 const &dims) const
// {
//   Sz3 pad_dims;
//   Sz3 pads;
//   for (Index ii = 0; ii < 3; ii++) {
//     if (L_ > 0) {
//       pad_dims[ii] = ((dims[ii] / (2 << L_)) + 1) * (2 << L_);
//       Index pad = (pad_dims[ii] - dims[ii]) / 2;
//       pads[ii] = pad; // std::make_pair(pad, pad);
//     } else {
//       pad_dims[ii] = dims[ii];
//       pads[ii] = 0;
//     }
//   }
//   Log::Print(FMT_STRING("Wavelet pad {} padded size {}"), pads[0], pad_dims[0]);
//   return {pad_dims, pads};
// }

auto Wavelets::inputDimensions() const -> InputDims
{
  return dims_;
}

auto Wavelets::outputDimensions() const -> OutputDims
{
  return dims_;
}

void Wavelets::encode_dim(Output &image, Index const dim, Index const level) const
{
  Index const sz = image.dimension(dim + 1) / (1 << level);
  Index const hsz = sz / 2;
  Sz2 start{0, 0};
  Sz2 end{dims_[0], sz};
  Log::Print<Log::Level::Debug>(FMT_STRING("Wavelet encode level: {} dim {} sz {} hsz {}"), level, dim, sz, hsz);

  auto encode_task = [&, sz, hsz, dim](Index const ii) {
    Cx2 temp(dims_[0], sz);
    for (Index ij = 0; ij < sz; ij++) {
      auto const line = image.chip(ii, dim1.at(dim) + 1).chip(ij, dim2.at(dim) + 1).slice(start, end);
      for (Index it = 0; it < hsz; it++) {
        temp.chip<1>(it).setZero();
        temp.chip<1>(it + hsz).setZero();
        Index f = 1;
        for (Index iw = 0; iw < N_; iw++) {
          Index const index = it * 2 + iw;
          if (index < sz) {
            temp.chip<1>(it) += line.chip<1>(index) * Cx(D_(iw));
            temp.chip<1>(it + hsz) += line.chip<1>(index) * Cx(D_(N_ - 1 - iw) * f);
          }
          f *= -1;
        }
      }
      image.chip(ii, dim1.at(dim) + 1).chip(ij, dim2.at(dim) + 1).slice(start, end) = temp;
    }
  };
  Threads::For(encode_task, sz, fmt::format(FMT_STRING("Wavelets Encode Dimension {} Level {}"), dim, level));
}

auto Wavelets::forward(Input const &x) const -> Output const &
{
  checkForward(x, "WaveletsOp");
  ws_ = x;
  for (Index il = 0; il < L_; il++) {
    for (Index dim = 0; dim < 3; dim++) {
      encode_dim(ws_, dim, il);
    }
  }
  return ws_;
}

void Wavelets::decode_dim(Input &image, Index const dim, Index const level) const
{
  Index const sz = image.dimension(dim + 1) / (1 << level);
  Index const hsz = sz / 2;
  Sz2 start{0, 0};
  Sz2 end{dims_[0], sz};
  Log::Print<Log::Level::Debug>(FMT_STRING("Wavelet decode level: {} dim {} sz {} hsz {}"), level, dim, sz, hsz);

  auto decode_task = [&, sz, hsz, dim](Index const ii) {
    Cx2 temp(dims_[0], sz);
    for (Index ij = 0; ij < sz; ij++) {
      temp.setZero();
      auto const line = image.chip(ii, dim1.at(dim) + 1).chip(ij, dim2.at(dim) + 1).slice(start, end);
      for (Index it = 0; it < hsz; it++) {
        Index const temp_index = it * 2;
        Index const N_2 = N_ / 2;
        for (Index iw = 0; iw < N_2; iw++) {
          Index const line_index = it - iw;
          if (line_index >= 0) {
            temp.chip<1>(temp_index) += line.chip<1>(line_index) * Cx(D_(iw * 2));
            temp.chip<1>(temp_index) += line.chip<1>(hsz + line_index) * Cx(D_((N_ - 1) - iw * 2));
            temp.chip<1>(temp_index + 1) += line.chip<1>(line_index) * Cx(D_(iw * 2 + 1));
            temp.chip<1>(temp_index + 1) -= line.chip<1>(hsz + line_index) * Cx(D_((N_ - 2) - iw * 2));
          }
        }
      }
      image.chip(ii, dim1.at(dim) + 1).chip(ij, dim2.at(dim) + 1).slice(start, end) = temp;
    }
  };
  Threads::For(decode_task, sz, fmt::format(FMT_STRING("Wavelets Decode Dimension {} Level {}"), dim, level));
}

auto Wavelets::adjoint(Output const &x) const -> Input const & 
{
  checkAdjoint(x, "WaveletsOp");
  ws_ = x;
  for (Index il = L_ - 1; il >= 0; il--) {
    for (Index dim = 0; dim < 3; dim++) {
      decode_dim(ws_, dim, il);
    }
  }
  return ws_;
}

} // namespace rl
