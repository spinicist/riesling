#include "types.hpp"

#include "algo/gs.hpp"
#include "basis/fourier.hpp"
#include "io/writer.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "algo/gs.hpp"

#include <complex>
#include <numbers>

using namespace std::literals::complex_literals;

int main_basis_ip(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> samples(parser, "S", "Number of samples (1)", {"samples", 's'}, 1);
  args::ValueFlag<Index> gap(parser, "G", "Gap before samples begin", {"gap", 'g'}, 0);
  args::ValueFlag<Index> tSamp(parser, "T", "Sample time (10μs)", {"tsamp", 't'}, 10);
  args::ValueFlag<Index> freq(parser, "F", "Fat frequency (-450 Hz)", {"freq", 'f'}, -450.f);
  args::Flag             fw(parser, "W", "Fat-Water not IP/OP", {"fw"});
  ParseCommand(parser, oname);

  auto const nS = samples.Get();
  auto const nG = gap.Get();
  auto const sampPhase = tSamp.Get() * 1e-6f * freq.Get() * 2.f * std::numbers::pi_v<float>;
  auto const startPhase = nG * sampPhase; 
  auto const endPhase = (nG + nS - 1) * sampPhase;

  rl::Log::Print("Gap {} Samples {} Max accumulated phase {} radians", nG, nS, endPhase);

  Eigen::VectorXcf water = Eigen::VectorXcf::Ones(nS);

  Eigen::VectorXcf const ph = Eigen::VectorXcf::LinSpaced(nS, startPhase * 1if, endPhase * 1if);
  Eigen::VectorXcf const fat = ph.array().exp();

  Eigen::MatrixXcf basis = Eigen::MatrixXcf::Zero(2, nS);
  if (fw) {
    basis.row(0) = water;
    basis.row(1) = fat;
  } else {
    basis.row(0) = (water + fat) / 2.f;
    basis.row(1) = (water - fat) / 2.f;
  }

  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, rl::Sz3{2, nS, 1}, basis.data(), rl::HD5::Dims::Basis);
  return EXIT_SUCCESS;
}
