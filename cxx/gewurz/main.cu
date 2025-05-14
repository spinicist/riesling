#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"

#include "dft.cuh"
#include "lsmr.cuh"
#include "recon.cuh"
#include "sense.hpp"

#include <args.hxx>

using namespace rl;

namespace {
std::unordered_map<int, Log::Display> levelMap{{0, Log::Display::None},
                                               {1, Log::Display::Ephemeral},
                                               {2, Log::Display::Low},
                                               {3, Log::Display::Mid},
                                               {4, Log::Display::High}};
} // namespace

auto ReadTrajectory(HD5::Reader &reader) -> DTensor<TDev, 3>
{
  Log::Print("gewurz", "Read trajectory");
  auto const shape = reader.dimensions();
  auto const nC = shape[0];
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

auto Preconditioner(DTensor<TDev, 3> const &T, rl::HD5::Shape<3> const mat) -> DTensor<TDev, 2>
{
  Log::Print("gewurz", "Preconditioner");
  auto const             nS = T.span.extent(1);
  auto const             nT = T.span.extent(2);
  DTensor<TDev, 2>       M(nS, nT);
  DTensor<CuCx<TDev>, 2> Mks(nS, nT);
  DTensor<CuCx<TDev>, 3> Mimg(mat[0], mat[1], mat[2]);
  thrust::fill(Mks.vec.begin(), Mks.vec.end(), CuCx<TDev>(1));

  gw::DFT::ThreeD dft{T.span};
  dft.adjoint(Mks.span, Mimg.span);
  dft.forward(Mimg.span, Mks.span);
  thrust::transform(thrust::cuda::par, Mks.vec.begin(), Mks.vec.end(), M.vec.begin(),
                    [] __device__(CuCx<TDev> x) { return TDev(1) / cuda::std::abs(x); });
  return M;
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
  std::transform(hKS.vec.begin(), hKS.vec.end(), hhKS.vec.begin(), ConvertToCx);
  DTensor<CuCx<TDev>, 3> ks(nC, nS, nT);
  thrust::copy(hhKS.vec.begin(), hhKS.vec.end(), ks.vec.begin());
  return ks;
}

auto ReadSENSE(std::string const &sname, rl::HD5::Shape<3> const mat, int const nC) -> DTensor<CuCx<TDev>, 4>
{
  auto                   hostS = gw::GetSENSE(sname, mat);
  HTensor<CuCx<TDev>, 4> hhS(mat[0], mat[1], mat[2], nC);
  DTensor<CuCx<TDev>, 4> S(mat[0], mat[1], mat[2], nC);
  std::transform(hostS.begin(), hostS.end(), hhS.vec.begin(), ConvertToCx);
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
  auto const             nI = hImg.span.extent(0);
  auto const             nJ = hImg.span.extent(1);
  auto const             nK = hImg.span.extent(2);
  auto                   Mop = gw::MulPacked<CuCx<TDev>, TDev, 3>{M.span};
  gw::Recon<NC>          dftp{S.span, T.span};
  DTensor<CuCx<TDev>, 3> img(nI, nJ, nK);
  Mop.forward(ks.span, ks.span);
  dftp.adjoint(ks.span, img.span);

  HTensor<CuCx<TDev>, 3> hhImg(nI, nJ, nK);
  thrust::copy(img.vec.begin(), img.vec.end(), hhImg.vec.begin());
  thrust::transform(hhImg.vec.begin(), hhImg.vec.end(), hImg.vec.begin(), ConvertFromCx);
}

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("GEWURZ");

  args::HelpFlag                   help(parser, "H", "Show this help message", {'h', "help"});
  args::MapFlag<int, Log::Display> verbosity(parser, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Display::Low);

  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> sname(parser, "FILE", "Input SENSE maps");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  parser.ParseCLI(argc, argv);
  if (verbosity) {
    Log::SetDisplayLevel(verbosity.Get());
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    Log::SetDisplayLevel(levelMap.at(std::atoi(env_p)));
  }
  Log::Print("gewurz", "Welcome!");

  HD5::Reader reader(iname.Get());
  // Info const  info = reader.readStruct<Info>(HD5::Keys::Info);
  auto const shape = reader.dimensions();
  auto const nC = shape[0];
  auto const nS = shape[1];
  auto const nT = shape[2];

  // HD5::Reader sread(sname.Get());

  HD5::Writer writer(oname.Get());
  // writer.writeStruct(HD5::Keys::Info, info);

  try {
    auto const mat = reader.readAttributeShape<3>(HD5::Keys::Trajectory, "matrix");
    auto       T = ReadTrajectory(reader);
    auto       M = Preconditioner(T, mat);
    auto       KS = ReadKS(reader);
    auto       S = ReadSENSE(sname.Get(), mat, nC);
    // Log::Print("gewurz", "Read SENSE maps");
    // HTensor<std::complex<float>, 4> sImgs(nC, mat[0], mat[1], mat[2]);
    // sread.readTo(sImgs.vec.data());

    HTensor<std::complex<float>, 3> img(mat[0], mat[1], mat[2]);
    switch (nC) {
    case 1: DoRecon<1>(KS, T, M, S, img); break;
    case 8: DoRecon<8>(KS, T, M, S, img); break;
    }
    writer.writeTensor("adjoint", HD5::Shape<3>{mat[0], mat[1], mat[2]}, img.vec.data(), {"i", "j", "k"});

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
