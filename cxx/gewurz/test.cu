#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"

#include "dft.cuh"
#include "lsmr.cuh"
#include "recon.cuh"
#include "sense.hpp"

#include <args.hxx>

#include <thrust/extrema.h>

using namespace rl;

namespace {
std::unordered_map<int, Log::Display> levelMap{{0, Log::Display::None},
                                               {1, Log::Display::Ephemeral},
                                               {2, Log::Display::Low},
                                               {3, Log::Display::Mid},
                                               {4, Log::Display::High}};
} // namespace

auto SetupCartesian(int M, HD5::Writer &writer) -> DTensor<TDev, 3>
{
  Log::Print("tests", "Setup trajectory");
  auto const nS = M;
  auto const nT = M * M;

  HTensor<float, 3> hT(3L, nS, nT);
  long int          it = 0;
  for (short ik = 0; ik < M; ik++) {
    TDev const kz = FLOAT_TO((ik - M / 2));
    for (short ij = 0; ij < M; ij++) {
      TDev const ky = FLOAT_TO((ij - M / 2));
      for (short ii = 0; ii < M; ii++) {
        TDev const kx = FLOAT_TO((ii - M / 2));
        hT.span(0, ii, it) = kx;
        hT.span(1, ii, it) = ky;
        hT.span(2, ii, it) = kz;
      }
      it++;
    }
  }
  writer.writeTensor("trajCart", HD5::Shape<3>{3, nS, nT}, hT.vec.data(), {"d", "s", "t"});
  HTensor<TDev, 3> hhT(3L, nS, nT);
  thrust::transform(hT.vec.begin(), hT.vec.end(), hhT.vec.begin(), ConvertTo);
  DTensor<TDev, 3> T(3L, nS, nT);
  thrust::copy(hhT.vec.begin(), hhT.vec.end(), T.vec.begin());
  return T;
}

auto Setup1(int const M, float const k, HD5::Writer &writer) -> DTensor<TDev, 3>
{
  Log::Print("tests", "Setup trajectory");
  auto const nS = 2;
  auto const nT = 1;

  HTensor<float, 3> hT(3L, nS, nT);
  hT.span(0, 0, 0) = 0;
  hT.span(1, 0, 0) = 0;
  hT.span(2, 0, 0) = -k;
  hT.span(0, 1, 0) = 0;
  hT.span(1, 1, 0) = 0;
  hT.span(2, 1, 0) = k;
  writer.writeTensor("traj1", HD5::Shape<3>{3, nS, nT}, hT.vec.data(), {"d", "s", "t"});
  HTensor<TDev, 3> hhT(3L, nS, nT);
  thrust::transform(hT.vec.begin(), hT.vec.end(), hhT.vec.begin(), ConvertTo);
  DTensor<TDev, 3> T(3L, nS, nT);
  thrust::copy(hhT.vec.begin(), hhT.vec.end(), T.vec.begin());
  return T;
}

void TestDFT(DTensor<TDev, 3> const &T, int const M, std::string const &prefix, HD5::Writer &writer)
{
  Log::Print("tests", "DFT");
  auto const             nS = T.span.extent(1);
  auto const             nT = T.span.extent(2);
  DTensor<CuCx<TDev>, 2> ks(nS, nT);
  DTensor<CuCx<TDev>, 3> img(M, M, M);
  thrust::fill(ks.vec.begin(), ks.vec.end(), CuCx<TDev>(1));
  gw::DFT::ThreeD dft{T.span};
  Log::Print("DFT", "|img| {} |ks| {}", gw::CuNorm(img.vec), gw::CuNorm(ks.vec));

  dft.adjoint(ks.span, img.span);
  Log::Print("DFT", "|img| {} |ks| {}", gw::CuNorm(img.vec), gw::CuNorm(ks.vec));
  HTensor<CuCx<TDev>, 3> hhImg(M, M, M);
  thrust::copy(img.vec.begin(), img.vec.end(), hhImg.vec.begin());
  HTensor<std::complex<float>, 3> hImg(M, M, M);
  thrust::transform(hhImg.vec.begin(), hhImg.vec.end(), hImg.vec.begin(), ToStdCx);
  writer.writeTensor(prefix + "img", HD5::Shape<3>{M, M, M}, hImg.vec.data(), {"i", "j", "k"});

  dft.forward(img.span, ks.span);
  Log::Print("DFT", "|img| {} |ks| {}", gw::CuNorm(img.vec), gw::CuNorm(ks.vec));
  HTensor<CuCx<TDev>, 2> hhKS(nS, nT);
  thrust::copy(ks.vec.begin(), ks.vec.end(), hhKS.vec.begin());
  HTensor<std::complex<float>, 2> hKS(nS, nT);
  thrust::transform(hhKS.vec.begin(), hhKS.vec.end(), hKS.vec.begin(), ToStdCx);
  writer.writeTensor(prefix + "ks", HD5::Shape<2>{nS, nT}, hKS.vec.data(), {"s", "t"});
}

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("GEWURZ");

  args::HelpFlag                   help(parser, "H", "Show this help message", {'h', "help"});
  args::MapFlag<int, Log::Display> verbosity(parser, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Display::Low);

  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<int>          M(parser, "M", "Matrix size", {'m', "mat"}, 6);
  args::ValueFlag<float>        k(parser, "N", "k", {'k', "k"}, 0);

  parser.ParseCLI(argc, argv);
  if (verbosity) {
    Log::SetDisplayLevel(verbosity.Get());
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    Log::SetDisplayLevel(levelMap.at(std::atoi(env_p)));
  }
  Log::Print("tests", "Welcome!");

  try {
    HD5::Writer writer(oname.Get());
    // auto        TC = SetupCartesian(M.Get(), writer);
    // TestDFT(TC, M.Get(), "cart", writer);
    auto T1 = Setup1(M.Get(), k.Get(), writer);
    TestDFT(T1, M.Get(), "1", writer);
    Log::Print("tests", "Finished");
  } catch (Log::Failure &f) {
    Log::Fail(f);
    Log::End();
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    Log::Fail(Log::Failure("None", "{}", e.what()));
    Log::End();
    return EXIT_FAILURE;
  }
  Log::End();
  return EXIT_SUCCESS;
}
