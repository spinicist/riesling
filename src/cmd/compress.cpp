#include "algo/decomp.h"
#include "compressor.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "rovir.hpp"
#include "tensorOps.hpp"
#include "types.h"

#include <Eigen/Eigenvalues>

using namespace rl;

int main_compress(args::Subparser &parser)
{
  CoreOpts core(parser);

  // One of the following must be set
  args::Flag pca(parser, "V", "Calculate PCA compression", {"pca"});
  args::Flag rovir(parser, "R", "Calculate ROVIR compression", {"rovir"});
  args::ValueFlag<std::string> ccFile(parser, "F", "Read compression matrix from file", {"cc-file"});

  // General options
  args::Flag save(parser, "S", "Save compression matrix to .h5 file", {"save"});
  args::ValueFlag<Index> channels(parser, "C", "Retain N channels (8)", {"channels"}, 8);
  args::ValueFlag<float> energy(parser, "E", "Retain fraction energy (overrides channels)", {"energy"}, -1.f);
  args::ValueFlag<Index> refVol(parser, "V", "Use this volume (default last)", {"vol"});
  args::ValueFlag<Index> lores(parser, "L", "Number of lores traces", {"lores"}, 0);

  // PCA Options
  args::ValueFlag<Sz2, Sz2Reader> pcaRead(parser, "R", "PCA Read Points (start, size)", {"pca-read"}, Sz2{0, 16});
  args::ValueFlag<Sz3, Sz3Reader> pcatraces(
    parser, "R", "PCA traces (start, size, stride)", {"pca-traces"}, Sz3{0, 1024, 4});

  ROVIROpts rovirOpts(parser);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto traj = reader.trajectory();
  auto const info = traj.info();
  Cx3 const ks = reader.noncartesian(ValOrLast(refVol, info.volumes));

  Eigen::MatrixXcf psi;
  if (pca) {
    Sz2 const read = pcaRead.Get();
    Sz3 const traces = pcatraces.Get();
    Index const maxRead = info.samples - read[0];
    Index const nread = (read[1] > maxRead) ? maxRead : read[1];
    if (traces[0] + traces[1] > info.traces) {
      Log::Fail(FMT_STRING("Requested end spoke {} is past end of file {}"), traces[0] + traces[1], info.traces);
    }
    Log::Print(FMT_STRING("Using {} read points, {} traces, {} stride"), nread, traces[1], traces[2]);
    Cx3 const ref =
      ks.slice(Sz3{0, read[0], traces[0]}, Sz3{info.channels, read[1], traces[1]}).stride(Sz3{1, 1, traces[2]});

    auto const pc = PCA(CollapseToConstMatrix(ref), channels.Get(), energy.Get());
    psi = pc.vecs;
  } else if (rovir) {
    psi = ROVIR(rovirOpts, traj, energy.Get(), channels.Get(), lores.Get(), ks);
  } else if (ccFile) {
    HD5::Reader matFile(ccFile.Get());
    psi = matFile.readMatrix<Eigen::MatrixXcf>(HD5::Keys::CompressionMatrix);
  } else {
    Log::Fail("Must specify PCA/ROVIR/load from file");
  }
  Compressor compressor{psi};
  Cx4 all_ks = info.noncartesianSeries();
  for (Index iv = 0; iv < info.volumes; iv++) {
    all_ks.chip<3>(iv) = reader.noncartesian(iv);
  }
  Info out_info = info;
  out_info.channels = compressor.out_channels();
  Cx4 out_ks = out_info.noncartesianSeries();
  compressor.compress(all_ks, out_ks);

  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "compressed"));
  writer.writeTrajectory(Trajectory(out_info, traj.points(), traj.frames()));
  writer.writeTensor(out_ks, HD5::Keys::Noncartesian);

  if (save) {
    HD5::Writer matfile(OutName(core.iname.Get(), core.oname.Get(), "ccmat"));
    matfile.writeMatrix(compressor.psi, HD5::Keys::CompressionMatrix);
  }
  return EXIT_SUCCESS;
}
