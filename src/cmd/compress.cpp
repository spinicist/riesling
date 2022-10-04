#include "algo/decomp.hpp"
#include "compressor.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "rovir.hpp"
#include "tensorOps.hpp"
#include "types.hpp"

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
  args::ValueFlag<Index> nRetain(parser, "C", "Retain N channels (8)", {"channels"}, 8);
  args::ValueFlag<float> energy(parser, "E", "Retain fraction energy (overrides channels)", {"energy"}, -1.f);
  args::ValueFlag<Index> refVol(parser, "V", "Use this volume (default first)", {"vol"}, 0);
  args::ValueFlag<Index> lores(parser, "L", "Number of lores traces", {"lores"}, 0);

  // PCA Options
  args::ValueFlag<Sz2, Sz2Reader> pcaRead(parser, "R", "PCA Samples (start, size)", {"pca-read"}, Sz2{0, 16});
  args::ValueFlag<Sz3, Sz3Reader> pcaTraces(
    parser, "R", "PCA Traces (start, size, stride)", {"pca-traces"}, Sz3{0, 1024, 4});

  ROVIROpts rovirOpts(parser);

  ParseCommand(parser, core.iname);

  HD5::Reader reader(core.iname.Get());
  Cx3 const ks = reader.readSlab<Cx3>(HD5::Keys::Noncartesian, refVol.Get());
  Index const channels = ks.dimension(0);
  Index const samples = ks.dimension(1);
  Index const traces = ks.dimension(2);
  Eigen::MatrixXcf psi;
  if (pca) {
    Index const maxRead = samples - pcaRead.Get()[0];
    Index const nread = (pcaRead.Get()[1] > maxRead) ? maxRead : pcaRead.Get()[1];
    if (pcaTraces.Get()[0] + pcaTraces.Get()[1] > traces) {
      Log::Fail(
        FMT_STRING("Requested end spoke {} is past end of file {}"), pcaTraces.Get()[0] + pcaTraces.Get()[1], traces);
    }
    Log::Print(FMT_STRING("Using {} read points, {} traces, {} stride"), nread, pcaTraces.Get()[1], pcaTraces.Get()[2]);
    Cx3 const ref =
      ks.slice(Sz3{0, pcaRead.Get()[0], pcaTraces.Get()[0]}, Sz3{channels, pcaRead.Get()[1], pcaTraces.Get()[1]})
        .stride(Sz3{1, 1, pcaTraces.Get()[2]});

    auto const pc = PCA(CollapseToConstMatrix(ref), nRetain.Get(), energy.Get());
    psi = pc.vecs;
  } else if (rovir) {
    psi = ROVIR(rovirOpts, Trajectory(reader), energy.Get(), nRetain.Get(), lores.Get(), ks);
  } else if (ccFile) {
    HD5::Reader matFile(ccFile.Get());
    psi = matFile.readMatrix<Eigen::MatrixXcf>(HD5::Keys::CompressionMatrix);
  } else {
    Log::Fail("Must specify PCA/ROVIR/load from file");
  }
  Compressor compressor{psi};
  Cx4 all_ks(reader.dimensions<4>(HD5::Keys::Noncartesian));
  Index const volumes = all_ks.dimension(3);
  for (Index iv = 0; iv < volumes; iv++) {
    all_ks.chip<3>(iv) = reader.readSlab<Cx3>(HD5::Keys::Noncartesian, iv);
  }
  Cx4 out_ks(AddFront(LastN<3>(all_ks.dimensions()), compressor.out_channels()));
  compressor.compress(all_ks, out_ks);

  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "compressed"));
  Trajectory(reader).write(writer);
  writer.writeTensor(out_ks, HD5::Keys::Noncartesian);

  if (save) {
    HD5::Writer matfile(OutName(core.iname.Get(), core.oname.Get(), "ccmat"));
    matfile.writeMatrix(compressor.psi, HD5::Keys::CompressionMatrix);
  }
  return EXIT_SUCCESS;
}
