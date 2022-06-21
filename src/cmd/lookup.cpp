#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "parse_args.h"
#include "tensorOps.h"
#include "threads.h"

int main_lookup(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::Positional<std::string> dname(parser, "DICT", "h5 file containing lookup dictionary");
  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::RieslingReader input(iname.Get());
  Cx5 const images = input.readTensor<Cx5>(HD5::Keys::Image);

  if (!dname) {
    throw args::Error("No basis file specified");
  }
  HD5::Reader dfile(dname.Get());
  R2 const basis = dfile.readTensor<R2>(HD5::Keys::Basis);
  R2 const dictionary = dfile.readTensor<R2>(HD5::Keys::Dictionary);
  R2 const parameters = dfile.readTensor<R2>(HD5::Keys::Parameters);
  R1 const norm = dfile.readTensor<R1>(HD5::Keys::Norm);

  R5 out_pars(
    parameters.dimension(0), images.dimension(1), images.dimension(2), images.dimension(3), images.dimension(4));
  out_pars.setZero();
  Cx5 pd(1, images.dimension(1), images.dimension(2), images.dimension(3), images.dimension(4));
  pd.setZero();

  Index const N = dictionary.dimension(0);
  if (parameters.dimension(1) != N) {
    Log::Fail(FMT_STRING("Dictionary has {} entries but parameters has {}"), N, parameters.dimension(1));
  }
  Log::Print(FMT_STRING("Dictionary has {} entries"), N);

  for (Index iv = 0; iv < images.dimension(4); iv++) {
    Log::Print(FMT_STRING("Processing volume {}"), iv);
    auto ztask = [&](Index const iz) {
      for (Index iy = 0; iy < images.dimension(2); iy++) {
        for (Index ix = 0; ix < images.dimension(1); ix++) {
          Cx1 const x = images.chip<4>(iv).chip<3>(iz).chip<2>(iy).chip<1>(ix);
          Index index = 0;
          Cx bestCorr{0.f, 0.f};
          float bestAbsCorr = 0;

          for (Index in = 0; in < N; in++) {
            R1 const atom = dictionary.chip<0>(in);
            Cx const corr = Dot(atom.cast<Cx>(), x);
            if (std::abs(corr) > bestAbsCorr) {
              bestAbsCorr = std::abs(corr);
              bestCorr = corr;
              index = in;
            }
          }
          out_pars.chip<4>(iv).chip<3>(iz).chip<2>(iy).chip<1>(ix) = parameters.chip<1>(index);
          pd(0, ix, iy, iz, iv) = bestCorr / norm(index);
        }
      }
    };
    Threads::For(ztask, images.dimension(3), "Lookup");
  }

  auto const fname = OutName(iname.Get(), oname.Get(), "lookup", "h5");
  HD5::Writer writer(fname);
  writer.writeTrajectory(input.trajectory());
  writer.writeTensor(out_pars, HD5::Keys::Parameters);
  writer.writeTensor(pd, HD5::Keys::ProtonDensity);

  return EXIT_SUCCESS;
}
