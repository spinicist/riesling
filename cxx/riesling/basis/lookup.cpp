#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensors.hpp"
#include "threads.hpp"
#include "basis/basis.hpp"


using namespace rl;

void main_lookup(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> dname(parser, "DICT", "h5 file containing lookup dictionary");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  ParseCommand(parser, iname, oname);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5 const   images = input.readTensor<Cx5>();

  if (!dname) { throw args::Error("No basis file specified"); }
  HD5::Reader dfile(dname.Get());
  Re2 const   dictionary = dfile.readTensor<Re2>(HD5::Keys::Dictionary);
  Re2 const   parameters = dfile.readTensor<Re2>(HD5::Keys::Parameters);
  Re1 const   norm = dfile.readTensor<Re1>(HD5::Keys::Norm);

  Re5 out_pars(parameters.dimension(0), images.dimension(1), images.dimension(2), images.dimension(3), images.dimension(4));
  out_pars.setZero();
  Cx4 pd(images.dimension(1), images.dimension(2), images.dimension(3), images.dimension(4));
  pd.setZero();

  Index const N = dictionary.dimension(1);
  if (parameters.dimension(1) != N) {
    Log::Fail("Dictionary has {} entries but parameters has {}", N, parameters.dimension(1));
  }
  Log::Print("Dictionary has {} entries", N);

  for (Index iv = 0; iv < images.dimension(4); iv++) {
    Log::Print("Processing volume {}", iv);
    auto ztask = [&](Index const iz) {
      for (Index iy = 0; iy < images.dimension(2); iy++) {
        for (Index ix = 0; ix < images.dimension(1); ix++) {
          Cx1 const x = images.chip<4>(iv).chip<3>(iz).chip<2>(iy).chip<1>(ix);
          Index     index = 0;
          Cx        bestCorr{0.f, 0.f};
          float     bestAbsCorr = 0;

          for (Index in = 0; in < N; in++) {
            Re1 const atom = dictionary.chip<1>(in);
            Cx const  corr = Dot(atom.cast<Cx>(), x);
            if (std::abs(corr) > bestAbsCorr) {
              bestAbsCorr = std::abs(corr);
              bestCorr = corr;
              index = in;
            }
          }
          out_pars.chip<4>(iv).chip<3>(iz).chip<2>(iy).chip<1>(ix) = parameters.chip<1>(index);
          pd(ix, iy, iz, iv) = bestCorr / norm(index);
        }
      }
    };
    Threads::For(ztask, images.dimension(3), "Lookup");
  }
  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Parameters, out_pars.dimensions(), out_pars.data(), {"p", "x", "y", "z", "t"});
  writer.writeTensor(HD5::Keys::ProtonDensity, pd.dimensions(), pd.data(), {"x", "y", "z", "t"});
}
