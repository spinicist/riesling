#include "inputs.hpp"

#include "rl/algo/decomp.hpp"
#include "rl/algo/stats.hpp"
#include "rl/compressor.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include <Eigen/Eigenvalues>

using namespace rl;

void main_compress(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<std::string> ccFile(parser, "F", "Read compression matrix from file", {"cc-file"});

  // General options
  args::ValueFlag<std::string> save(parser, "S", "Save compression matrix to .h5 file", {"save"});
  args::ValueFlag<Index>       nRetain(parser, "C", "Retain N channels (8)", {"channels", 'n'}, 8);
  args::ValueFlag<float>       energy(parser, "E", "Retain fraction energy (overrides channels)", {"energy"}, -1.f);
  args::ValueFlag<Index>       refVol(parser, "V", "Use this volume (default first)", {"vol"}, 0);
  args::ValueFlag<Index>       lores(parser, "L", "Number of lores traces", {"lores"}, 0);

  // PCA Options
  args::ValueFlag<Sz2, SzReader<2>> pcaRead(parser, "R", "PCA Samples (start, size)", {"pca-samp"}, Sz2{0, 16});
  args::ValueFlag<Sz3, SzReader<3>> pcaTraces(parser, "R", "PCA Traces (start, size, stride)", {"pca-traces"}, Sz3{0, 1024, 1});
  args::ValueFlag<Sz2, SzReader<2>> pcaSlices(parser, "R", "PCA Slices (start, size)", {"pca-slices"}, Sz2{0, 1});

  ParseCommand(parser, iname, oname);
  auto const cmd = parser.GetCommand().Name();

  HD5::Reader reader(iname.Get());
  Info const  info = reader.readStruct<Info>(HD5::Keys::Info);
  Trajectory  traj(reader, info.voxel_size);
  Cx4 const   ks = reader.readSlab<Cx4>(HD5::Keys::Data, {{4, refVol.Get()}});
  Index const channels = ks.dimension(0);
  Index const samples = ks.dimension(1);
  Index const traces = ks.dimension(2);

  if (nRetain.Get() > channels) {
    throw Log::Failure("Compress", "Data had {} channels requested {}", channels, nRetain.Get());
  };
  Index const toKeep = nRetain.Get() > 0 ? nRetain.Get() : channels;

  Eigen::MatrixXcf psi;
  if (ccFile) {
    HD5::Reader matFile(ccFile.Get());
    psi = matFile.readMatrix<Eigen::MatrixXcf>(HD5::Keys::CompressionMatrix);
    if (psi.rows() != channels) { throw Log::Failure(cmd, "Incompatible compression matrix"); }
    if (nRetain && nRetain.Get() < psi.cols()) { psi = psi.leftCols(nRetain.Get()); }
  } else {
    Index const maxRead = samples - pcaRead.Get()[0];
    Index const nread = (pcaRead.Get()[1] > maxRead) ? maxRead : pcaRead.Get()[1];
    Index const maxTrace = traces - pcaTraces.Get()[0];
    Index const nTrace = (pcaTraces.Get()[1] > maxTrace) ? maxTrace : pcaTraces.Get()[1];
    Cx4 const   ref = ks.slice(Sz4{0, pcaRead.Get()[0], pcaTraces.Get()[0], pcaSlices.Get()[0]},
                               Sz4{channels, nread, nTrace, pcaSlices.Get()[1]})
                      .stride(Sz4{1, 1, pcaTraces.Get()[2], 1});
    auto const eig = Eig<Cx>(Covariance(CollapseToConstMatrix(ref)));
    auto const nR = energy ? CountCumulativeBelow(eig.V, energy.Get()) : toKeep;
    psi = eig.P.leftCols(nR);
  }
  Compressor  compressor{psi};
  Cx5 const   uncompressed = reader.readTensor<Cx5>();
  Cx5         compressed(AddFront(LastN<4>(uncompressed.dimensions()), psi.cols()));
  Index const volumes = uncompressed.dimension(4);
  for (Index iv = 0; iv < volumes; iv++) {
    compressed.chip<4>(iv) = compressor.compress(Cx4(uncompressed.chip<4>(iv)));
  }

  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  traj.write(writer);
  writer.writeTensor(HD5::Keys::Data, ToArray(compressed.dimensions()), compressed.data(), HD5::Dims::Noncartesian);

  if (save) {
    HD5::Writer matfile(save.Get());
    matfile.writeTensor(HD5::Keys::CompressionMatrix, {compressor.psi.rows(), compressor.psi.cols()}, compressor.psi.data(),
                        HD5::DNames<2>{"oc", "ic"});
  }
  Log::Print(cmd, "Finished");
}
