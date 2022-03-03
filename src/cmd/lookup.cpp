#include "types.h"

#include "io/io.h"
#include "log.h"
#include "parse_args.h"
#include "tensorOps.h"
#include "threads.h"

int main_lookup(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::Positional<std::string> dname(parser, "DICT", "h5 file containing lookup dictionary");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string> oftype(
    parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");
  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 const images = input.readTensor<Cx5>("image");

  if (!dname) {
    throw args::Error("No basis file specified");
  }
  HD5::Reader dfile(dname.Get());
  R2 const basis = dfile.readTensor<R2>("basis");
  R2 const dictionary = dfile.readTensor<R2>("dictionary");
  R2 const parameters = dfile.readTensor<R2>("parameters");
  R2 const norm = dfile.readTensor<R2>(HD5::Keys::Norm);

  R5 out_pars(
    parameters.dimension(1),
    images.dimension(1),
    images.dimension(2),
    images.dimension(3),
    images.dimension(4));
  out_pars.setZero();
  Cx5 pd(1, images.dimension(1), images.dimension(2), images.dimension(3), images.dimension(4));
  pd.setZero();

  Index const N = dictionary.dimension(0);
  if (parameters.dimension(0) != N) {
    Log::Fail(
      FMT_STRING("Dictionary has {} entries but parameters has {}"), N, parameters.dimension(0));
  }
  Log::Print(FMT_STRING("Dictionary has {} entries"), N);

  for (Index iv = 0; iv < images.dimension(4); iv++) {
    Log::Print(FMT_STRING("Processing volume {}"), iv);
    auto ztask = [&](Index const lo, Index const hi, Index const ti) {
      for (Index iz = lo; iz < hi; iz++) {
        Log::Progress(iz, lo, hi);
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
            out_pars.chip<4>(iv).chip<3>(iz).chip<2>(iy).chip<1>(ix) = parameters.chip<0>(index);
            pd(0, ix, iy, iz, iv) = bestCorr / norm(index, 0);
          }
        }
      }
    };
    Threads::RangeFor(ztask, images.dimension(3));
  }

  auto const fname = OutName(iname.Get(), oname.Get(), "dict", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(out_pars, "parameters");
  writer.writeTensor(pd, "pd");

  return EXIT_SUCCESS;
}
