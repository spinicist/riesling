#include "io/hd5.hpp"
#include "log.hpp"
#include "inputs.hpp"
#include "tensors.hpp"
#include "types.hpp"

using namespace rl;

void main_noisify(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<float>        σ(parser, "S", "Noise standard deviation", {"std"}, 1.f);

  ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get());
  Cx5         ks = reader.readTensor<Cx5>();

  Cx5 noise(ks.dimensions());
  noise.setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
  ks += noise * noise.constant(σ.Get());

  HD5::Writer writer(oname.Get());
  Info const  info = reader.readInfo();
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
  Trajectory traj(reader, info.voxel_size);
  traj.write(writer);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
