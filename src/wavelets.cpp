#include "threads.h"
#include "wavelets.h"
#include <fmt/format.h>
#include <iostream>
#include <vector>

// Once the first chip is done the Tensor will be 2D and oh look I've gone cross-eyed
namespace {
std::vector<long> const dim1 = {1, 2, 0};
std::vector<long> const dim2 = {1, 0, 0};
} // namespace

Wavelets::Wavelets(long const N, long const L, Log &log)
    : N_{N}
    , L_{L}
    , log_{log}
{

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
    D_.setValues({0.32580343f,
                  1.01094572f,
                  0.89220014f,
                  -0.03957503f,
                  -0.26450717f,
                  0.0436163f,
                  0.0465036f,
                  -0.01498699f});
    break;
  default:
    Log::Fail("Asked for co-efficients that have not been implemented");
  }
  D_ = D_ / static_cast<float>(M_SQRT2); // Get scaling correct
  log_.info(FMT_STRING("Wavelets: N={} L={}"), N_, L_);
}

std::tuple<Sz3, Sz3> Wavelets::pad_setup(Sz3 const &dims) const
{
  Sz3 pad_dims;
  Sz3 pads;
  for (long ii = 0; ii < 3; ii++) {
    if (L_ > 0) {
      pad_dims[ii] = ((dims[ii] / (2 << L_)) + 1) * (2 << L_);
      long pad = (pad_dims[ii] - dims[ii]) / 2;
      pads[ii] = pad; // std::make_pair(pad, pad);
    } else {
      pad_dims[ii] = dims[ii];
      pads[ii] = 0;
    }
  }
  log_.info(FMT_STRING("Wavelet pad {} padded size {}"), pads[0], pad_dims[0]);
  return {pad_dims, pads};
}

void Wavelets::pad(Cx3 const &src, Cx3 &dest)
{
  Sz3 pad_start;
  for (long ii = 0; ii < 3; ii++) {
    pad_start[ii] = (dest.dimension(ii) - src.dimension(ii)) / 2;
  }
  dest.slice(pad_start, src.dimensions()) = src;
}
void Wavelets::unpad(Cx3 const &src, Cx3 &dest)
{
  Sz3 pad_start;
  for (long ii = 0; ii < 3; ii++) {
    pad_start[ii] = (src.dimension(ii) - dest.dimension(ii)) / 2;
  }
  dest = src.slice(pad_start, dest.dimensions());
}

void Wavelets::encode_dim(Cx3 &image, long const dim, long const level)
{
  long const sz = image.dimension(dim) / (1 << level);
  long const hsz = sz / 2;
  Sz1 start{0};
  Sz1 end{sz};
  log_.info(FMT_STRING("Wavelet encode level: {} dim {} sz {} hsz {}"), level, dim, sz, hsz);

  auto encode_task = [&, sz, hsz, dim](long const lo, long const hi) {
    for (long ii = lo; ii < hi; ii++) {
      for (long ij = 0; ij < sz; ij++) {
        Cx1 temp(sz);
        Cx1 const line = image.chip(ii, dim1.at(dim)).chip(ij, dim2.at(dim)).slice(start, end);
        for (long it = 0; it < hsz; it++) {
          temp(it) = std::complex<float>{0.f, 0.f};
          temp(it + hsz) = std::complex<float>{0.f, 0.f};
          long f = 1;
          for (long iw = 0; iw < N_; iw++) {
            long const index = it * 2 + iw;
            if (index < sz) {
              temp(it) += line(index) * D_(iw);
              temp(it + hsz) += line(index) * (D_(N_ - 1 - iw) * f);
            }
            f *= -1;
          }
        }
        image.chip(ii, dim1.at(dim)).chip(ij, dim2.at(dim)).slice(start, end) = temp;
      }
    }
  };
  Threads::RangeFor(encode_task, sz);
}

void Wavelets::encode(Cx3 &image)
{

  for (long il = 0; il < L_; il++) {
    for (long dim = 0; dim < 3; dim++) {
      encode_dim(image, dim, il);
    }
  }
}

void Wavelets::decode_dim(Cx3 &image, long const dim, long const level)
{
  long const sz = image.dimension(dim) / (1 << level);
  long const hsz = sz / 2;
  Sz1 start{0};
  Sz1 end{sz};
  log_.info(FMT_STRING("Wavelet decode level: {} dim {} sz {} hsz {}"), level, dim, sz, hsz);

  auto decode_task = [&, sz, hsz, dim](long const lo, long const hi) {
    for (long ii = lo; ii < hi; ii++) {
      for (long ij = 0; ij < sz; ij++) {
        Cx1 temp(sz);
        temp.setZero();
        Eigen::TensorRef<Cx1> line =
            image.chip(ii, dim1.at(dim)).chip(ij, dim2.at(dim)).slice(start, end);
        for (long it = 0; it < hsz; it++) {
          long const temp_index = it * 2;
          long const N_2 = N_ / 2;
          for (long iw = 0; iw < N_2; iw++) {
            long const line_index = it - iw;
            if (line_index >= 0) {
              temp(temp_index) += line(line_index) * D_(iw * 2);
              temp(temp_index) += line(hsz + line_index) * D_((N_ - 1) - iw * 2);
              temp(temp_index + 1) += line(line_index) * D_(iw * 2 + 1);
              temp(temp_index + 1) -= line(hsz + line_index) * D_((N_ - 2) - iw * 2);
            }
          }
        }
        image.chip(ii, dim1.at(dim)).chip(ij, dim2.at(dim)).slice(start, end) = temp;
      }
    }
  };
  Threads::RangeFor(decode_task, sz);
}

void Wavelets::decode(Cx3 &image)
{
  for (long il = L_ - 1; il >= 0; il--) {
    for (long dim = 0; dim < 3; dim++) {
      decode_dim(image, dim, il);
    }
  }
}
