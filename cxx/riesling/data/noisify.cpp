#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
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
  writer.writeInfo(reader.readInfo());
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
  Re3 const traj = reader.readTensor<Re3>(HD5::Keys::Trajectory);
  writer.writeTensor(HD5::Keys::Trajectory, traj.dimensions(), traj.data(), HD5::Dims::Trajectory);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
