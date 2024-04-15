#include "types.hpp"

#include "io/writer.hpp"
#include "log.hpp"
#include "parse_args.hpp"

#include <Eigen/Householder>
#include <complex>
#include <numbers>

using namespace std::literals::complex_literals;

void main_ipop_basis(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index>     samples(parser, "S", "Number of samples (1)", {"samples", 's'}, 1);
  args::ValueFlag<Index>     gap(parser, "G", "Gap before samples begin", {"gap", 'g'}, 0);
  args::ValueFlag<Index>     tSamp(parser, "T", "Sample time (10μs)", {"tsamp", 't'}, 10);
  args::ValueFlagList<float> freqs(parser, "F", "Fat frequencies (-450 Hz)", {"freq", 'f'}, {440.f});
  ParseCommand(parser, oname);

  auto const       nS = samples.Get();
  auto const       nG = gap.Get();
  auto const      &fs = freqs.Get();
  auto const       nF = (Index)fs.size();
  Eigen::MatrixXcf basis = Eigen::MatrixXcf::Zero(nS, nF + 1);
  basis.col(0).setOnes();

  for (Index ii = 0; ii < nF; ii++) {
    auto const sampPhase = tSamp.Get() * 1e-6f * fs[ii] * 2.f * std::numbers::pi_v<float>;
    auto const startPhase = nG * sampPhase;
    auto const endPhase = (nG + nS - 1) * sampPhase;

    rl::Log::Print("Gap {} Samples {} Max accumulated phase {} radians", nG, nS, endPhase);
    basis.col(ii + 1) = Eigen::VectorXcf::LinSpaced(nS, startPhase * 1if, endPhase * 1if).array().exp();
  }

  auto const             h = basis.householderQr();
  Eigen::MatrixXcf const Q = Eigen::MatrixXcf(h.householderQ()).leftCols(nF + 1) * std::sqrt(nS);
  Eigen::MatrixXcf const R = Eigen::MatrixXcf(h.matrixQR().topRows(nF + 1).triangularView<Eigen::Upper>()) / std::sqrt(nS);
  rl::Log::Print("Orthog check\n{}", fmt::streamed(Q.adjoint() * Q / nS));
  rl::HD5::Writer writer(oname.Get());
  Eigen::MatrixXcf const tb = Q.transpose();
  writer.writeTensor(rl::HD5::Keys::Basis, rl::Sz3{nF + 1L, nS, 1}, tb.data(), rl::HD5::Dims::Basis);
  writer.writeMatrix(R, "R");
  rl::Log::Print("Finished {}", parser.GetCommand().Name());
}
