#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/nufft.hpp"
#include "precon.hpp"
#include "sys/threads.hpp"

using namespace rl;

void main_nufft(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridArgs<3> gridOpts(parser);
  PreconOpts  preOpts(parser);
  LsqOpts     lsqOpts(parser);

  args::Flag fwd(parser, "", "Apply forward operator", {'f', "fwd"});
  args::Flag adj(parser, "", "Apply adjoint operator", {'a', "adj"});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreOpts.iname.Get());

  Trajectory traj(reader, reader.readInfo().voxel_size, gridOpts.matrix.Get());
  auto const basis = LoadBasis(coreOpts.basisFile.Get());

  auto const shape = reader.dimensions();

  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeInfo(reader.readInfo());
  traj.write(writer);

  if (fwd) {
    auto const cart = reader.readTensor<Cx6>();
    auto const nC = shape[1];
    auto const nS = 1;
    auto const nT = shape[5];
    auto const A = TOps::NUFFTAll(gridOpts.Get(), traj, nC, nS, nT, basis.get());
    auto const noncart = A->forward(cart);
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    auto const nC = shape[0];
    auto const nS = shape[3];
    auto const nT = shape[4];
    auto const A = TOps::NUFFTAll(gridOpts.Get(), traj, nC, nS, nT, basis.get());
    if (adj) {
      auto const cart = A->adjoint(noncart);
      writer.writeTensor(HD5::Keys::Data, cart.dimensions(), cart.data(), HD5::Dims::Channels);
    } else {
      auto const M = MakeKspacePre(traj, nC, nS, nT, basis.get(), preOpts.type.Get(), preOpts.bias.Get());
      LSMR const lsmr{A, M, nullptr, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};
      auto const c = lsmr.run(CollapseToConstVector(noncart));
      writer.writeTensor(HD5::Keys::Data, A->ishape, c.data(), HD5::Dims::Channels);
    }
  }
  Log::Print(cmd, "Finished");
}
