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

struct CuCxFToCuCxH
{
  __host__ __device__ CuCxH operator()(CuCxF const z) const { return CuCxH(__float2bfloat16(z.real()), __float2bfloat16(z.imag())); }
};

struct CuCxHToCuCxF
{
  __host__ __device__ CuCxF operator()(CuCxH const z) const { return CuCxF(__bfloat162float(z.real()), __bfloat162float(z.imag())); }
};

struct FloatToHalf
{
  __host__ __device__ __nv_bfloat16 operator()(float const f) const
  {
    return __float2bfloat16(f);
  }
};
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
  Info const  info = reader.readInfo();
  auto const  shape = reader.dimensions();
  Index const nC = shape[0];
  Index const nS = shape[1];
  Index const nT = shape[2];

  HD5::Writer writer(oname.Get());
  writer.writeInfo(info);

  try {
  Log::Print("gewurz", "Read trajectory");
  HTensor<float, 3> hT(3L, nS, nT);
  reader.readTo(hT.vec.data(), HD5::Keys::Trajectory);
  auto const        mat = reader.readAttributeSz<3>(HD5::Keys::Trajectory, "matrix");
  HTensor<__nv_bfloat16, 3> hhT(3L, nS, nT);
  thrust::transform(hT.vec.begin(), hT.vec.end(), hhT.vec.begin(), FloatToHalf());
  DTensor<__nv_bfloat16, 3> T(3L, nS, nT);
  thrust::copy(hhT.vec.begin(), hhT.vec.end(), T.vec.begin());
  

  // Log::Print("gewurz", "Preconditioner");
  // DTensor<__nv_bfloat16, 2>  M(nS, nT);
  // DTensor<CuCxH, 2> Mks(nS, nT);
  // DTensor<CuCxH, 3> Mimg(mat[0], mat[1], mat[2]);
  // thrust::fill(Mks.vec.begin(), Mks.vec.end(), CuCxH(1));

  // gw::DFT::ThreeD dft{T.span};
  // dft.adjoint(Mks.span, Mimg.span);
  // dft.forward(Mimg.span, Mks.span);
  // thrust::transform(thrust::cuda::par, Mks.vec.begin(), Mks.vec.end(), M.vec.begin(),
  //                   [] __device__(CuCxH x) { return __nv_bfloat16(1) / cuda::std::abs(x); });
  // gw::MulPacked<CuCxH, __nv_bfloat16, 3> Mop{M.span};

  Log::Print("gewurz", "Read k-space");
  HTensor<CuCxF, 3> hKS(nC, nS, nT);
  reader.readTo((Cx *)hKS.vec.data());
  HTensor<CuCxH, 3> hhKS(nC, nS, nT);
  std::transform(hKS.vec.begin(), hKS.vec.end(), hhKS.vec.begin(), CuCxFToCuCxH());
  DTensor<CuCxH, 3> ks(nC, nS, nT);
  thrust::copy(hhKS.vec.begin(), hhKS.vec.end(), ks.vec.begin());

  Log::Print("gewurz", "Recon");
  gw::DFT::ThreeDPacked<8> dftp{T.span};
  HTensor<CuCxF, 4>        hImgs(nC, mat[0], mat[1], mat[2]);
  HTensor<CuCxH, 4>        hhImgs(nC, mat[0], mat[1], mat[2]);
  DTensor<CuCxH, 4>        imgs(nC, mat[0], mat[1], mat[2]);
  // Mop.forward(ks.span, ks.span);
  dftp.adjoint(ks.span, imgs.span);
  thrust::copy(imgs.vec.begin(), imgs.vec.end(), hhImgs.vec.begin());
  std::transform(hhImgs.vec.begin(), hhImgs.vec.end(), hImgs.vec.begin(), CuCxHToCuCxF());
  writer.writeTensor("adjoint", Sz4{nC, mat[0], mat[1], mat[2]}, (Cx *)hImgs.vec.data(), {"channel", "i", "j", "k"});

  // gw::LSMR<CuCxH, 4, 3> lsmr{&dftp, &Mop};
  // thrust::copy(hhKS.vec.begin(), hhKS.vec.end(), ks.vec.begin());
  // lsmr.run(ks, imgs);
  // thrust::copy(imgs.vec.begin(), imgs.vec.end(), hhImgs.vec.begin());
  // std::transform(hhImgs.vec.begin(), hhImgs.vec.end(), hImgs.vec.begin(), CuCxHToCuCxF());
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
