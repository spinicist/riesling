#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_noisify(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<float>        σ(parser, "S", "Noise standard deviation (1.0)", {"std", 's'}, 1.f);

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());

  Cx5 ks = reader.readTensor<Cx5>();

  Cx5 noise(ks.dimensions());
  noise.setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
  ks += noise * noise.constant(σ.Get());

  HD5::Writer writer(oname.Get());
  Info const  info = reader.readStruct<Info>(HD5::Keys::Info);
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), reader.readDNames<5>());
  if (reader.exists(HD5::Keys::Trajectory)) {
    Trajectory traj(reader, info.voxel_size);
    traj.write(writer);
  }
  Log::Print(cmd, "Finished");
}
