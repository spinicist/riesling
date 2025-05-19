#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"

#include "../algo/lsmr.cuh"
#include "../args.hpp"
#include "../op/dft.cuh"
#include "../op/recon.cuh"
#include "../sense.hpp"
#include "info.hpp"

#include <thrust/extrema.h>

using namespace rl;

auto ReadTrajectory(HD5::Reader &reader) -> DTensor<TDev, 3>
{
  Log::Print("gewurz", "Read trajectory");
  auto const shape = reader.dimensions("trajectory");
  auto const nS = shape[1];
  auto const nT = shape[2];

  HTensor<float, 3> hT(3L, nS, nT);
  reader.readTo(hT.vec.data(), HD5::Keys::Trajectory);
  HTensor<TDev, 3> hhT(3L, nS, nT);
  thrust::transform(hT.vec.begin(), hT.vec.end(), hhT.vec.begin(), ConvertTo);
  DTensor<TDev, 3> T(3L, nS, nT);
  thrust::copy(hhT.vec.begin(), hhT.vec.end(), T.vec.begin());
  return T;
}

void WriteTrajectory(DTensor<TDev, 3> const &T, HD5::Shape<3> const mat, HD5::Writer &writer)
{
  Log::Print("gewurz", "Write trajectory");
  auto const nS = T.span.extent(1);
  auto const nT = T.span.extent(2);

  HTensor<TDev, 3> hhT(3L, nS, nT);
  thrust::copy(T.vec.begin(), T.vec.end(), hhT.vec.begin());
  HTensor<float, 3> hT(3L, nS, nT);
  thrust::transform(hhT.vec.begin(), hhT.vec.end(), hT.vec.begin(), ConvertFrom);

  writer.writeTensor(HD5::Keys::Trajectory, HD5::Shape<3>(3L, nS, nT), hT.vec.data(), HD5::Dims::Trajectory);
  writer.writeAttribute(HD5::Keys::Trajectory, "matrix", mat);
}

void DoTest(DTensor<TDev, 3> const &T, rl::HD5::Shape<3> const mat)
{
  Log::Print("gewurz", "Test");
  auto const             nS = T.span.extent(1);
  auto const             nT = T.span.extent(2);
  UTensor<CuCx<TDev>, 2> Mks(nS, nT);
  UTensor<CuCx<TDev>, 3> Mimg(mat[0], mat[1], mat[2]);

  thrust::fill(Mks.vec.begin(), Mks.vec.end(), CuCx<TDev>(1));
  // for (int it = 0; it < nT; it++) {
  //   Mks.span(nS - 1, it) = CuCx<TDev>(1);
  // }

  HD5::Writer                     debug("test.h5");
  HTensor<CuCx<TDev>, 2>          hks(nS, nT);
  HTensor<std::complex<float>, 2> hhks(nS, nT);
  thrust::copy(Mks.vec.begin(), Mks.vec.end(), hks.vec.begin());
  thrust::transform(hks.vec.begin(), hks.vec.end(), hhks.vec.begin(), ToStdCx);
  debug.writeTensor("Mks1", HD5::Shape<2>{nS, nT}, hhks.vec.data(), {"s", "t"});

  gw::DFT::ThreeD dft{T.span};
  dft.adjoint(Mks.span, Mimg.span);
  Log::Print("Test", "|img| {} |ks| {}", gw::CuNorm(Mimg.vec), gw::CuNorm(Mks.vec));

  HTensor<CuCx<TDev>, 3>          himg(mat[0], mat[1], mat[2]);
  HTensor<std::complex<float>, 3> hhimg(mat[0], mat[1], mat[2]);
  thrust::copy(Mimg.vec.begin(), Mimg.vec.end(), himg.vec.begin());
  thrust::transform(himg.vec.begin(), himg.vec.end(), hhimg.vec.begin(), ToStdCx);
  debug.writeTensor("Mimg", HD5::Shape<3>{mat[0], mat[1], mat[2]}, hhimg.vec.data(), {"i", "j", "k"});

  dft.forward(Mimg.span, Mks.span);
  Log::Print("Test", "|img| {} |ks| {}", gw::CuNorm(Mimg.vec), gw::CuNorm(Mks.vec));

  thrust::copy(Mks.vec.begin(), Mks.vec.end(), hks.vec.begin());
  thrust::transform(hks.vec.begin(), hks.vec.end(), hhks.vec.begin(), ToStdCx);
  debug.writeTensor("Mks2", HD5::Shape<2>{nS, nT}, hhks.vec.data(), {"s", "t"});
}

auto Preconditioner(DTensor<TDev, 3> const &T, int const nI, int const nJ, int const nK) -> DTensor<TDev, 2>
{
  Log::Print("gewurz", "Preconditioner");
  auto const       nS = T.span.extent(1);
  auto const       nT = T.span.extent(2);
  DTensor<TDev, 2> M(nS, nT);
  thrust::fill(M.vec.begin(), M.vec.end(), TDev(1));

  DTensor<CuCx<TDev>, 2> Mks(nS, nT);
  DTensor<CuCx<TDev>, 3> Mimg(nI, nJ, nK);
  thrust::fill(Mks.vec.begin(), Mks.vec.end(), CuCx<TDev>(1));

  HD5::Writer debug("debug.h5");

  gw::DFT::ThreeD dft{T.span};
  Log::Print("Precon", "|img| {} |ks| {}", gw::CuNorm(Mimg.vec), gw::CuNorm(Mks.vec));
  dft.adjoint(Mks.span, Mimg.span);
  Log::Print("Precon", "|img| {} |ks| {}", gw::CuNorm(Mimg.vec), gw::CuNorm(Mks.vec));
  dft.forward(Mimg.span, Mks.span);
  Log::Print("Precon", "|img| {} |ks| {}", gw::CuNorm(Mimg.vec), gw::CuNorm(Mks.vec));
  float const λ = 0.0f;
  thrust::transform(thrust::cuda::par, Mks.vec.begin(), Mks.vec.end(), M.vec.begin(),
                    [λ] __device__(CuCx<TDev> x) { return TDev(1 + λ) / (cuda::std::abs(x) + λ); });
  auto const mm = thrust::minmax_element(thrust::cuda::par, M.vec.begin(), M.vec.end());
  TDev const min = *(mm.first);
  TDev const max = *(mm.second);
  Log::Print("Precon", "|M| {} Min {} Max {}", gw::CuNorm(M.vec), min, max);
  return M;
}

auto ReadImage(rl::HD5::Reader &reader) -> DTensor<CuCx<TDev>, 4>
{
  Log::Print("gewurz", "Read k-space");
  auto const shape = reader.dimensions();
  auto const nC = shape[3];
  auto const nI = shape[0];
  auto const nJ = shape[1];
  auto const nK = shape[2];
  fmt::print(stderr, "{} {} {} {}\n", nI, nJ, nK, nC);
  HTensor<std::complex<float>, 4> himg(nI, nJ, nK, nC);
  reader.readTo(himg.vec.data());
  fmt::print(stderr, "|himg| {}\n", gw::CuNorm(himg.vec));
  HTensor<CuCx<TDev>, 4> hhimg(nI, nJ, nK, nC);
  std::transform(himg.vec.begin(), himg.vec.end(), hhimg.vec.begin(), FromStdCx);
  fmt::print(stderr, "|hhimg| {}\n", gw::CuNorm(hhimg.vec));
  DTensor<CuCx<TDev>, 4> img(nI, nJ, nK, nC);
  thrust::copy(hhimg.vec.begin(), hhimg.vec.end(), img.vec.begin());
  fmt::print(stderr, "|img| {}\n", gw::CuNorm(img.vec));
  return img;
}

auto ReadKS(rl::HD5::Reader &reader) -> DTensor<CuCx<TDev>, 3>
{
  Log::Print("gewurz", "Read k-space");
  auto const                      shape = reader.dimensions();
  auto const                      nC = shape[0];
  auto const                      nS = shape[1];
  auto const                      nT = shape[2];
  HTensor<std::complex<float>, 3> hKS(nC, nS, nT);
  reader.readTo(hKS.vec.data());
  HTensor<CuCx<TDev>, 3> hhKS(nC, nS, nT);
  std::transform(hKS.vec.begin(), hKS.vec.end(), hhKS.vec.begin(), FromStdCx);
  DTensor<CuCx<TDev>, 3> ks(nC, nS, nT);
  thrust::copy(hhKS.vec.begin(), hhKS.vec.end(), ks.vec.begin());
  return ks;
}

auto ReadSENSE(std::string const &sname, rl::HD5::Shape<3> const mat, int const nC) -> DTensor<CuCx<TDev>, 4>
{
  auto                   hostS = gw::GetSENSE(sname, mat);
  HTensor<CuCx<TDev>, 4> hhS(mat[0], mat[1], mat[2], nC);
  DTensor<CuCx<TDev>, 4> S(mat[0], mat[1], mat[2], nC);
  std::transform(hostS.begin(), hostS.end(), hhS.vec.begin(), FromStdCx);
  thrust::copy(hhS.vec.begin(), hhS.vec.end(), S.vec.begin());
  return S;
}

template <int NC> void DoRecon(DTensor<CuCx<TDev>, 3> const    &ks,
                               DTensor<TDev, 3> const          &T,
                               DTensor<TDev, 2> const          &M,
                               DTensor<CuCx<TDev>, 4> const    &S,
                               HTensor<std::complex<float>, 3> &hImg)

{
  Log::Print("gewurz", "Recon");
  auto const    nI = hImg.span.extent(0);
  auto const    nJ = hImg.span.extent(1);
  auto const    nK = hImg.span.extent(2);
  auto          Minv = gw::MulPacked<CuCx<TDev>, TDev, 3>{M.span};
  gw::Recon<NC> A{S.span, T.span};

  DTensor<CuCx<TDev>, 3>     img(nI, nJ, nK);
  gw::LSMR<CuCx<TDev>, 3, 3> lsmr{&A, &Minv};
  lsmr.run(ks, img);
  HTensor<CuCx<TDev>, 3> hhImg(nI, nJ, nK);
  thrust::copy(img.vec.begin(), img.vec.end(), hhImg.vec.begin());
  thrust::transform(hhImg.vec.begin(), hhImg.vec.end(), hImg.vec.begin(), ToStdCx);
}

template <int NC>
void DoForwardDFT(DTensor<CuCx<TDev>, 4> const &imgs, DTensor<TDev, 3> const &T, HTensor<std::complex<float>, 3> &hKS)
{
  Log::Print("gewurz", "Recon");
  auto const                nC = imgs.span.extent(3);
  auto const                nI = imgs.span.extent(0);
  auto const                nJ = imgs.span.extent(1);
  auto const                nK = imgs.span.extent(2);
  auto const                nS = hKS.span.extent(1);
  auto const                nT = hKS.span.extent(2);
  gw::DFT::ThreeDPacked<NC> A{T.span};
  DTensor<CuCx<TDev>, 3>    ks(nC, nS, nT);
  A.forward(imgs.span, ks.span);
  fmt::print(stderr, "|img| {} |ks| {}\n", gw::CuNorm(imgs.vec), gw::CuNorm(ks.vec));
  HTensor<CuCx<TDev>, 3> hhKS(nC, nS, nT);
  thrust::copy(ks.vec.begin(), ks.vec.end(), hhKS.vec.begin());
  thrust::transform(hhKS.vec.begin(), hhKS.vec.end(), hKS.vec.begin(), ToStdCx);
  fmt::print(stderr, "|hKs| {} |hhKS| {}\n", gw::CuNorm(hKS.vec), gw::CuNorm(hhKS.vec));
}

template <int NC>
void DoAdjointDFT(DTensor<CuCx<TDev>, 3> const &ks, DTensor<TDev, 3> const &T, HTensor<std::complex<float>, 4> &hImg)
{
  Log::Print("gewurz", "Recon");
  auto const nC = hImg.span.extent(3);
  auto const nI = hImg.span.extent(0);
  auto const nJ = hImg.span.extent(1);
  auto const nK = hImg.span.extent(2);

  gw::DFT::ThreeDPacked<NC> A{T.span};
  DTensor<CuCx<TDev>, 4>    imgs(nI, nJ, nK, nC);
  A.adjoint(ks.span, imgs.span);

  HTensor<CuCx<TDev>, 4> hhImg(nI, nJ, nK, nC);
  thrust::copy(imgs.vec.begin(), imgs.vec.end(), hhImg.vec.begin());
  thrust::transform(hhImg.vec.begin(), hhImg.vec.end(), hImg.vec.begin(), FromStdCx);
}

template <int NC> void DoInverseDFT(DTensor<CuCx<TDev>, 3> const    &ks,
                                    DTensor<TDev, 3> const          &T,
                                    bool const                       precon,
                                    HTensor<std::complex<float>, 4> &hImg)
{
  Log::Print("gewurz", "Recon");
  auto const nC = hImg.span.extent(3);
  auto const nI = hImg.span.extent(0);
  auto const nJ = hImg.span.extent(1);
  auto const nK = hImg.span.extent(2);

  DTensor<CuCx<TDev>, 4> imgs(nI, nJ, nK, nC);
  gw::DFT::ThreeDPacked<NC>          A{T.span};
  
  if (precon) {
    auto const                         W = Preconditioner(T, nI, nJ, nK);
    gw::MulPacked<CuCx<TDev>, TDev, 3> Minv{W.span};
    gw::LSMR lsmr{&A, &Minv};
    lsmr.run(ks, imgs);
  } else {
    gw::LSMR lsmr{&A};
    lsmr.run(ks, imgs);
  }

  HTensor<CuCx<TDev>, 4> hhImg(nI, nJ, nK, nC);
  thrust::copy(imgs.vec.begin(), imgs.vec.end(), hhImg.vec.begin());
  thrust::transform(hhImg.vec.begin(), hhImg.vec.end(), hImg.vec.begin(), FromStdCx);
}

void main_dft(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::Flag                    adj(parser, "A", "Adjoint only", {'a', "adj"});
  args::Flag                    fwd(parser, "F", "Forward", {'f', "fwd"});
  args::Flag                    noM(parser, "M", "No preconditioning", {'p', "nop"});

  ParseCommand(parser, iname, oname);
  Log::Print("DFT", "Welcome!");

  HD5::Reader reader(iname.Get());
  auto const  mat = reader.readAttributeShape<3>(HD5::Keys::Trajectory, "matrix");
  auto const  T = ReadTrajectory(reader);
  auto const  info = reader.readStruct<gw::Info>(HD5::Keys::Info);
  HD5::Writer writer(oname.Get());
  WriteTrajectory(T, mat, writer);
  writer.writeStruct(HD5::Keys::Info, info);

  if (fwd) {
    auto const shape = reader.dimensions();
    auto const nC = shape[3];
    auto const nS = T.span.extent(1);
    auto const nT = T.span.extent(2);
    auto       img = ReadImage(reader);
    fmt::print(stderr, "|img| {}\n", gw::CuNorm(img.vec));
    HTensor<std::complex<float>, 3> ks(nC, nS, nT);
    switch (nC) {
    case 1: DoForwardDFT<1>(img, T, ks); break;
    case 2: DoForwardDFT<2>(img, T, ks); break;
    case 4: DoForwardDFT<4>(img, T, ks); break;
    case 8: DoForwardDFT<8>(img, T, ks); break;
    default: throw(Log::Failure("DFT", "Unsupported number of channels {}", nC));
    }
    fmt::print(stderr, "|ks| {}\n", gw::CuNorm(ks.vec));
    writer.writeTensor("data", HD5::Shape<5>{nC, nS, nT, 1, 1}, ks.vec.data(), {"channel", "sample", "trace", "slab", "t"});
  } else {
    auto const                      shape = reader.dimensions();
    auto const                      nC = shape[0];
    auto                            KS = ReadKS(reader);
    HTensor<std::complex<float>, 4> img(mat[0], mat[1], mat[2], nC);
    if (adj) {
      switch (nC) {
      case 1: DoAdjointDFT<1>(KS, T, img); break;
      case 2: DoAdjointDFT<2>(KS, T, img); break;
      case 4: DoAdjointDFT<4>(KS, T, img); break;
      case 8: DoAdjointDFT<8>(KS, T, img); break;
      default: throw(Log::Failure("DFT", "Unsupported number of channels {}", nC));
      }
    } else {
      switch (nC) {
      case 1: DoInverseDFT<1>(KS, T, !noM, img); break;
      case 2: DoInverseDFT<2>(KS, T, !noM, img); break;
      case 4: DoInverseDFT<4>(KS, T, !noM, img); break;
      case 8: DoInverseDFT<8>(KS, T, !noM, img); break;
      default: throw(Log::Failure("DFT", "Unsupported number of channels {}", nC));
      }
    }
    writer.writeTensor("data", HD5::Shape<4>{mat[0], mat[1], mat[2], nC}, img.vec.data(), {"i", "j", "k", "channel"});
  }

  Log::Print("DFT", "Finished");
}
