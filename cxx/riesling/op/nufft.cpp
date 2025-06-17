#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/loopify.hpp"
#include "rl/op/nufft.hpp"
#include "rl/precon.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

template <int ND> void run_nufft(args::Subparser &parser)
{
  CoreArgs<ND> coreArgs(parser);
  GridArgs<ND> gridArgs(parser);
  PreconArgs   preArgs(parser);
  LSMRArgs     lsqOpts(parser);

  args::Flag fwd(parser, "", "Apply forward operator", {'f', "fwd"});
  args::Flag adj(parser, "", "Apply adjoint operator", {'a', "adj"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());

  TrajectoryN<ND> traj(reader, reader.readStruct<Info>(HD5::Keys::Info).voxel_size.head<ND>(), coreArgs.matrix.Get());
  auto const      basis = LoadBasis(coreArgs.basisFile.Get());

  auto const shape = reader.dimensions();

  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info));
  traj.write(writer);

  if (fwd) {
    auto const cart = reader.readTensor<Cx6>();
    Index      nS;
    if constexpr (ND == 2) {
      nS = shape[2];
    } else {
      nS = 1;
    }
    auto const nC = shape[3];
    auto const nT = shape[5];
    auto const nufft = TOps::NUFFT<ND>::Make(gridArgs.Get(), traj, nC, basis.get());
    auto const A = Loopify<ND>(nufft, nS, nT);
    auto const noncart = A->forward(cart);
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    auto const nC = shape[0];
    auto const nS = shape[3];
    auto const nT = shape[4];
    auto const nufft = TOps::NUFFT<ND>::Make(gridArgs.Get(), traj, nC, basis.get());
    auto const A = Loopify<ND>(nufft, nS, nT);
    if (adj) {
      auto const cart = A->adjoint(noncart);
      writer.writeTensor(HD5::Keys::Data, cart.dimensions(), cart.data(), HD5::Dims::Channels);
    } else {
      auto const M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, nC, Sz2{nS, nT});
      LSMR const lsmr{A, M, nullptr, lsqOpts.Get()};
      auto const c = lsmr.run(CollapseToConstVector(noncart));
      writer.writeTensor(HD5::Keys::Data, A->ishape, c.data(), HD5::Dims::Channels);
    }
  }
  Log::Print(cmd, "Finished");
}

void main_nufft(args::Subparser &parser) { run_nufft<3>(parser); }
void main_nufft2(args::Subparser &parser) { run_nufft<2>(parser); }