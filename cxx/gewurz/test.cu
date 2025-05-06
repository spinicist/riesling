#include "rl/log.hpp"

#include "dft.cuh"
#include "lsmr.cuh"

#include <args.hxx>

using namespace rl;

namespace {
std::unordered_map<int, Log::Display> levelMap{{0, Log::Display::None},
                                               {1, Log::Display::Ephemeral},
                                               {2, Log::Display::Low},
                                               {3, Log::Display::Mid},
                                               {4, Log::Display::High}};
}

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("GEWURZTestraminer");

  args::HelpFlag                   help(parser, "H", "Show this help message", {'h', "help"});
  args::MapFlag<int, Log::Display> verbosity(parser, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Display::Low);

  parser.ParseCLI(argc, argv);
  if (verbosity) {
    Log::SetDisplayLevel(verbosity.Get());
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    Log::SetDisplayLevel(levelMap.at(std::atoi(env_p)));
  }
  Log::Print("gewurztraminer", "Welcome!");

  Index const nC = 2;
  Index const nS = 4;
  Index const nT = 3;

  Log::Print("gewurztraminer", "Setup Trajectory");
  HTensor<CuReal, 3> hT(3, nS, nT);
  for (int it = 0; it < nT; it++) {
    for (int is = 0; is < nS; is++) {
        hT.span(it % 3, is, it) = is;
    }
  }
  DTensor<CuReal, 3> T(3L, nS, nT);
  thrust::copy(hT.vec.begin(), hT.vec.end(), T.vec.data());

  std::array<int, 3> mat{8, 8, 8};

  Log::Print("gewurztraminer", "Poor man's SDC");
  DTensor<CuReal, 2> M(nS, nT);
  DTensor<CuCx<THost>, 2>  Mks(nS, nT);
  DTensor<CuCx<THost>, 3>  Mimg(mat[0], mat[1], mat[2]);
  thrust::fill(Mks.vec.begin(), Mks.vec.end(), 1.f);

  gw::DFT::ThreeD dft{T.span};
  dft.adjoint(Mks.span, Mimg.span);
  dft.forward(Mimg.span, Mks.span);
  thrust::transform(thrust::cuda::par, Mks.vec.begin(), Mks.vec.end(), M.vec.begin(),
                    [] __device__(CuCx<THost> x) { return (CuReal)1 / cuda::std::abs(x); });
  fmt::print(stderr, "|Mks| {} |M| {}\n", gw::CuNorm(Mks.vec), gw::CuNorm(M.vec));
  gw::MulPacked<CuCx<THost>, CuReal, 3> Mop{M.span};

  Log::Print("gewurztraminer", "Setup k-space");
  HTensor<CuCx<THost>, 3> hKS(nC, nS, nT);
  DTensor<CuCx<THost>, 3> ks(nC, nS, nT);;
  thrust::fill(hKS.vec.begin(), hKS.vec.end(), CuCx<THost>(1.f));
  thrust::copy(hKS.vec.begin(), hKS.vec.end(), ks.vec.begin());

  Log::Print("gewurztraminer", "Recon");
  gw::DFT::ThreeDPacked dftp{T.span};
  gw::LSMR<CuCx<THost>, 4, 3> lsmr{&dftp};
  HTensor<CuCx<THost>, 4>     hImgs(nC, mat[0], mat[1], mat[2]);
  DTensor<CuCx<THost>, 4>     imgs(nC, mat[0], mat[1], mat[2]);
  fmt::print(stderr, "Before |ks| {} |imgs| {}\n", gw::CuNorm(ks.vec), gw::CuNorm(imgs.vec));
  dftp.adjoint(ks.span, imgs.span);
  fmt::print(stderr, "After |ks| {} |imgs| {}\n", gw::CuNorm(ks.vec), gw::CuNorm(imgs.vec));
  lsmr.run(ks, imgs);
  thrust::copy(imgs.vec.begin(), imgs.vec.end(), hImgs.vec.begin());
  return EXIT_SUCCESS;
}
