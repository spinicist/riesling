#include "types.h"

#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "tensorOps.h"
#include "threads.h"

int main_basis_dict(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::Positional<std::string> dname(parser, "DICT", "h5 file containing lookup dictionary");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string> oftype(
      parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");
  Log log = ParseCommand(parser);

  HD5::Reader input(iname.Get(), log);
  HD5::Reader dict(dname.Get(), log);

  auto const info = input.readInfo();

  Cx5 const images = input.readBasisImages();

  R2 const basis = input.readBasis();
  R2 const check = dict.readBasis();

  R2 const dictionary = dict.readRealMatrix("dictionary");
  R2 const parameters = dict.readRealMatrix("parameters");

  WriteBasisVolumes(
      images, basis, false, info, iname.Get(), oname.Get(), "basis", "nii", log);

  R5 out_pars(
      parameters.dimension(1),
      images.dimension(1),
      images.dimension(2),
      images.dimension(3),
      images.dimension(4));
  out_pars.setZero();

  long const N = dictionary.dimension(0);
  if (parameters.dimension(0) != N) {
    Log::Fail("Dictionary has {} entries but parameters has {}", N, parameters.dimension(0));
  }

  for (long iv = 0; iv < images.dimension(4); iv++) {
    log.info("Processing volume {}", iv);
    auto ztask = [&](long const lo, long const hi, long const ti) {
      for (long iz = lo; iz < hi; iz++) {
        log.progress(iz, lo, hi);
        for (long iy = 0; iy < images.dimension(2); iy++) {
          for (long ix = 0; ix < images.dimension(1); ix++) {
            Cx1 proj = images.chip(iv, 4).chip(iz, 3).chip(iy, 2).chip(ix, 1);
            float norm = Norm(proj);
            proj = proj / proj.constant(norm);
            long index = 0;
            float dist = 0;
            for (long in = 0; in < N; in++) {
              R1 const dict_proj = dictionary.chip(in, 0);
              float const test_dist =
                  std::abs(Dot(dict_proj.cast<Cx>(), proj / proj.constant(Norm(proj))));
              if (test_dist > dist) {
                dist = test_dist;
                index = in;
              }
            }
            out_pars.chip(iv, 4).chip(iz, 3).chip(iy, 2).chip(ix, 1) = parameters.chip(index, 0);
            out_pars(0, ix, iy, iz, iv) = norm / out_pars(0, ix, iy, iz, iv); // Scale Mz_ss to M0
          }
        }
      }
    };
    Threads::RangeFor(ztask, images.dimension(3));
  }

  auto const ext = oftype.Get();
  if (ext.compare("h5") == 0) {
    auto const fname = OutName(iname.Get(), oname.Get(), "dict", ext);
    HD5::Writer writer(fname, log);
    writer.writeReal5(out_pars, "parameters");
  } else {
    for (long iv = 0; iv < out_pars.dimension(4); iv++) {
      auto const fname =
          OutName(iname.Get(), oname.Get(), fmt::format("parameter-{:02d}", iv), ext);
      R4 const p = FirstToLast4(out_pars.chip(iv, 4));
      WriteNifti(info, p, fname, log);
    }
  }

  return EXIT_SUCCESS;
}
