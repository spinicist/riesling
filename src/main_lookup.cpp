#include "types.h"

#include "io.h"
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
  Log log = ParseCommand(parser);

  HD5::Reader input(iname.Get(), log);
  HD5::Reader dict(dname.Get(), log);

  Cx5 const images = input.readBasisImages();

  R2 const basis = input.readBasis();
  R2 const check = dict.readBasis();

  R2 const dictionary = dict.readRealMatrix("dictionary");
  R2 const parameters = dict.readRealMatrix("parameters");
  R2 const Mz_ss = dict.readRealMatrix("Mz_ss");

  R5 out_pars(
    parameters.dimension(1),
    images.dimension(1),
    images.dimension(2),
    images.dimension(3),
    images.dimension(4));
  out_pars.setZero();
  Cx4 pd(images.dimension(1), images.dimension(2), images.dimension(3), images.dimension(4));
  pd.setZero();

  long const N = dictionary.dimension(0);
  if (parameters.dimension(0) != N) {
    Log::Fail("Dictionary has {} entries but parameters has {}", N, parameters.dimension(0));
  }
  log.info(FMT_STRING("Dictionary has {} entries"), N);

  Cx1 const basis_ss = basis.chip<0>(0).cast<Cx>();
  for (long iv = 0; iv < images.dimension(4); iv++) {
    log.info("Processing volume {}", iv);
    auto ztask = [&](long const lo, long const hi, long const ti) {
      for (long iz = lo; iz < hi; iz++) {
        log.progress(iz, lo, hi);
        for (long iy = 0; iy < images.dimension(2); iy++) {
          for (long ix = 0; ix < images.dimension(1); ix++) {
            Cx1 const proj = images.chip<4>(iv).chip<3>(iz).chip<2>(iy).chip<1>(ix);
            long index = 0;
            float bestDot = 0;
            for (long in = 0; in < N; in++) {
              R1 const atom = dictionary.chip<0>(in);
              float const dot = std::abs(Dot(atom.cast<Cx>(), proj));
              if (dot > bestDot) {
                bestDot = dot;
                index = in;
              }
            }
            out_pars.chip<4>(iv).chip<3>(iz).chip<2>(iy).chip<1>(ix) = parameters.chip<0>(index);
            pd(ix, iy, iz, iv) = Dot(basis_ss, proj) / Mz_ss(index, 0); // Scale Mz_ss to M0
          }
        }
      }
    };
    Threads::RangeFor(ztask, images.dimension(3));
  }

  auto const fname = OutName(iname.Get(), oname.Get(), "dict", "h5");
  HD5::Writer writer(fname, log);
  writer.writeTensor(out_pars, "parameters");
  writer.writeTensor(pd, "pd");

  return EXIT_SUCCESS;
}
