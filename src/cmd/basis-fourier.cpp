#include "types.hpp"

#include "algo/gs.hpp"
#include "basis/fourier.hpp"
#include "io/writer.hpp"
#include "log.hpp"
#include "parse_args.hpp"

#include <complex>
#include <numbers>

using namespace std::literals::complex_literals;

int main_basis_fourier(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> N(parser, "N", "Number of Fourier harmonics (4)", {"N", 'N'}, 4);
  args::ValueFlag<Index> samples(parser, "S", "Number of samples (1)", {"samples", 's'}, 1);
  args::ValueFlag<Index> traces(parser, "T", "Number of traces (1)", {"traces", 't'}, 1);
  args::ValueFlag<Index> osamp(parser, "O", "Oversampling (1)", {"osamp", 'o'}, 1.f);
  args::Flag             method(parser, "P", "Use alternate method", {"method", 'm'});
  args::Flag             ortho(parser, "O", "Gram-Schmidt orthonormalization", {"ortho"});
  ParseCommand(parser, oname);

  if (method) {
    rl::FourierBasis fb(N.Get(), samples.Get(), traces.Get(), osamp.Get());
    fb.writeTo(oname.Get());
  } else {
    Eigen::MatrixXcf basis = Eigen::MatrixXcf::Zero(samples.Get(), 2 * N + 1);
    basis.col(0).setConstant(1.f);
    for (Index ii = 0; ii < N; ii++) {
      Eigen::VectorXcf const ph = Eigen::VectorXcf::LinSpaced(samples.Get(), 0.f, std::numbers::pi_v<float> * (1 + ii) * 1if);
      Eigen::VectorXcf const eph = ph.array().exp();
      basis.col(2 * ii + 1) = eph;
      basis.col(2 * ii + 2) = eph.conjugate();
    }

    Eigen::MatrixXcf const fbasis = ortho ? rl::GramSchmidt(basis, true).transpose() : basis.transpose();

    rl::HD5::Writer writer(oname.Get());
    writer.writeTensor(rl::HD5::Keys::Basis, rl::Sz3{2 * N.Get() + 1, samples.Get(), 1}, fbasis.data(), rl::HD5::Dims::Basis);
  }
  return EXIT_SUCCESS;
}
