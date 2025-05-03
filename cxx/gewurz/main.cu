#include "rl/io/hd5.hpp"
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
  args::ArgumentParser parser("GEWURZ");

  args::HelpFlag                   help(parser, "H", "Show this help message", {'h', "help"});
  args::MapFlag<int, Log::Display> verbosity(parser, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Display::Low);

  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  parser.ParseCLI(argc, argv);
  if (verbosity) {
    Log::SetDisplayLevel(verbosity.Get());
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    Log::SetDisplayLevel(levelMap.at(std::atoi(env_p)));
  }
  Log::Print("gewurz", "Welcome!");

  HD5::Reader reader(iname.Get());
  Info const  info = reader.readInfo();
  auto const  shape = reader.dimensions();
  Index const nC = shape[0];
  Index const nS = shape[1];
  Index const nT = shape[2];

  Log::Print("gewurz", "Read trajectory");
  HTensor<CuReal, 3> hT(3, nS, nT);
  reader.readTo(hT.vec.data(), HD5::Keys::Trajectory);
  auto const         mat = reader.readAttributeSz<3>(HD5::Keys::Trajectory, "matrix");
  DTensor<CuReal, 3> T(3L, nS, nT);
  thrust::copy(hT.vec.begin(), hT.vec.end(), T.vec.data());

  Log::Print("gewurz", "Poor man's SDC");
  DTensor<CuReal, 2> M(nS, nT);
  DTensor<CuCxF, 2>  Mks(nS, nT);
  DTensor<CuCxF, 3>  Mimg(mat[0], mat[1], mat[2]);
  thrust::fill(Mks.vec.begin(), Mks.vec.end(), 1.f);

  gw::DFT::ThreeD dft{T.span};
  dft.adjoint(Mks.span, Mimg.span);
  dft.forward(Mimg.span, Mks.span);
  thrust::transform(thrust::cuda::par, Mks.vec.begin(), Mks.vec.end(), M.vec.begin(),
                    [] __device__(CuCxF x) { return (CuReal)1 / cuda::std::abs(x); });

  Log::Print("gewurz", "Read k-space");
  HTensor<CuCxF, 3> hKS(nC, nS, nT);
  DTensor<CuCxF, 3> ks(nC, nS, nT);
  reader.readTo((Cx *)hKS.vec.data());
  thrust::copy(hKS.vec.begin(), hKS.vec.end(), ks.vec.begin());
  gw::MulPacked<CuCxF, CuReal, 3> Mop{M.span};
  Log::Print("gewurz", "Apply SDC");
  Mop.forward(ks.span, ks.span);
  HTensor<CuCxF, 4>     hImgs(nC, mat[0], mat[1], mat[2]);
  DTensor<CuCxF, 4>     imgs(nC, mat[0], mat[1], mat[2]);
  gw::DFT::ThreeDPacked dftp{T.span};
  Log::Print("gewurz", "Recon");
  dftp.adjoint(ks.span, imgs.span);
  thrust::copy(imgs.vec.begin(), imgs.vec.end(), hImgs.vec.begin());
  HD5::Writer writer(oname.Get());
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, Sz4{nC, mat[0], mat[1], mat[2]}, (Cx *)hImgs.vec.data(), {"channel", "i", "j", "k"});

  return EXIT_SUCCESS;
}
