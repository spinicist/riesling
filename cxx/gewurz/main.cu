#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"

#include "dft.cuh"
#include "dft2.cuh"
#include "lsmr.cuh"

#include <args.hxx>

using namespace rl;

namespace {
std::unordered_map<int, Log::Display> levelMap{{0, Log::Display::None},
                                               {1, Log::Display::Ephemeral},
                                               {2, Log::Display::Low},
                                               {3, Log::Display::Mid},
                                               {4, Log::Display::High}};
} // namespace

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
  Info const  info = reader.readStruct<Info>(HD5::Keys::Info);
  auto const  shape = reader.dimensions();
  Index const nC = shape[0];
  Index const nS = shape[1];
  Index const nT = shape[2];

  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);

  try {
    Log::Print("gewurz", "Read trajectory");
    HTensor<THost, 3> hT(3L, nS, nT);
    reader.readTo(hT.vec.data(), HD5::Keys::Trajectory);
    auto const       mat = reader.readAttributeArray<3>(HD5::Keys::Trajectory, "matrix");
    HTensor<TDev, 3> hhT(3L, nS, nT);
    thrust::transform(hT.vec.begin(), hT.vec.end(), hhT.vec.begin(), ConvertTo());
    DTensor<TDev, 3> T(3L, nS, nT);
    thrust::copy(hhT.vec.begin(), hhT.vec.end(), T.vec.begin());

    Log::Print("gewurz", "Preconditioner");
    DTensor<TDev, 2>       M(nS, nT);
    DTensor<CuCx<TDev>, 2> Mks(nS, nT);
    DTensor<CuCx<TDev>, 3> Mimg(mat[0], mat[1], mat[2]);
    thrust::fill(Mks.vec.begin(), Mks.vec.end(), CuCx<TDev>(1));

    gw::DFT::ThreeD dft{T.span};
    gw::DFT::ThreeD2 dft2{T.span}; // This one uses CUB
    fmt::print(stderr, "Before |Mks| {} |Mimg| {}\n", FLOAT_FROM(gw::CuNorm(Mks.vec)), FLOAT_FROM(gw::CuNorm(Mimg.vec)));
    dft2.adjoint(Mks.span, Mimg.span);
    fmt::print(stderr, "Middle |Mks| {} |Mimg| {}\n", FLOAT_FROM(gw::CuNorm(Mks.vec)),
               FLOAT_FROM(gw::CuNorm(Mimg.vec)));
    dft2.forward(Mimg.span, Mks.span);
    fmt::print(stderr, "After |Mks| {} |Mimg| {}\n", FLOAT_FROM(gw::CuNorm(Mks.vec)),
               FLOAT_FROM(gw::CuNorm(Mimg.vec)));
    thrust::transform(thrust::cuda::par, Mks.vec.begin(), Mks.vec.end(), M.vec.begin(),
                      [] __device__(CuCx<TDev> x) { return TDev(1) / cuda::std::abs(x); });
    gw::MulPacked<CuCx<TDev>, TDev, 3> Mop{M.span};

    Log::Print("gewurz", "Read k-space");
    HTensor<CuCx<THost>, 3> hKS(nC, nS, nT);
    reader.readTo((Cx *)hKS.vec.data());
    HTensor<CuCx<TDev>, 3> hhKS(nC, nS, nT);
    std::transform(hKS.vec.begin(), hKS.vec.end(), hhKS.vec.begin(), ConvertToCx());
    DTensor<CuCx<TDev>, 3> ks(nC, nS, nT);
    thrust::copy(hhKS.vec.begin(), hhKS.vec.end(), ks.vec.begin());

    Log::Print("gewurz", "Recon");
    gw::DFT::ThreeDPacked2<8> dftp{T.span};
    HTensor<CuCx<THost>, 4>  hImgs(nC, mat[0], mat[1], mat[2]);
    HTensor<CuCx<TDev>, 4>   hhImgs(nC, mat[0], mat[1], mat[2]);
    DTensor<CuCx<TDev>, 4>   imgs(nC, mat[0], mat[1], mat[2]);
    Mop.forward(ks.span, ks.span);
    dftp.adjoint(ks.span, imgs.span);
    thrust::copy(imgs.vec.begin(), imgs.vec.end(), hhImgs.vec.begin());
    std::transform(hhImgs.vec.begin(), hhImgs.vec.end(), hImgs.vec.begin(), ConvertFromCx());
    writer.writeTensor("adjoint", Sz4{nC, mat[0], mat[1], mat[2]}, (Cx *)hImgs.vec.data(), {"channel", "i", "j", "k"});

    // gw::LSMR<CuCx<TDev>, 4, 3> lsmr{&dftp, &Mop};
    // thrust::copy(hhKS.vec.begin(), hhKS.vec.end(), ks.vec.begin());
    // lsmr.run(ks, imgs);
    // thrust::copy(imgs.vec.begin(), imgs.vec.end(), hhImgs.vec.begin());
    // std::transform(hhImgs.vec.begin(), hhImgs.vec.end(), hImgs.vec.begin(), ConvertFromCx());
    // writer.writeTensor("inverse", Sz4{nC, mat[0], mat[1], mat[2]}, (Cx *)hImgs.vec.data(), {"channel", "i", "j", "k"});
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
