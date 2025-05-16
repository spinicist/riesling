#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/nufft.hpp"
#include "rl/op/loopify.hpp"
#include "rl/precon.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_nufft(args::Subparser &parser)
{
  CoreArgs<3> coreArgs(parser);
  GridArgs<3> gridArgs(parser);
  PreconArgs  preArgs(parser);
  LSMRArgs    lsqOpts(parser);

  args::Flag fwd(parser, "", "Apply forward operator", {'f', "fwd"});
  args::Flag adj(parser, "", "Apply adjoint operator", {'a', "adj"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());

  Trajectory traj(reader, reader.readStruct<Info>(HD5::Keys::Info).voxel_size, coreArgs.matrix.Get());
  auto const basis = LoadBasis(coreArgs.basisFile.Get());

  auto const shape = reader.dimensions();

  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info));
  traj.write(writer);

  if (fwd) {
    auto const cart = reader.readTensor<Cx6>();
    auto const nC = shape[3];
    auto const nS = 1;
    auto const nT = shape[5];
    auto const nufft = TOps::NUFFT<3>::Make(gridArgs.Get(), traj, nC, basis.get());
    auto const A = Loopify<TOps::NUFFT<3>>(nufft, nS, nT);
    auto const noncart = A->forward(cart);
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    auto const nC = shape[0];
    auto const nS = shape[3];
    auto const nT = shape[4];
    auto const nufft = TOps::NUFFT<3>::Make(gridArgs.Get(), traj, nC, basis.get());
    auto const A = Loopify<TOps::NUFFT<3>>(nufft, nS, nT);
    auto const M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, nC, nS, nT);
    if (adj) {
      auto const cart = A->adjoint(M->forward(noncart));
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

auto Loopify2(typename TOps::NUFFT<2>::Ptr op, Index const nSlice, Index const nTime) -> TOps::TOp<Cx, 6, 5>::Ptr
{
  TOps::TOp<Cx, 5, 4>::Ptr sliceLoop = TOps::MakeLoop<2, 3>(op, nSlice);
  TOps::TOp<Cx, 6, 5>::Ptr timeLoop = TOps::MakeLoop<5, 4>(sliceLoop, nTime);
  return timeLoop;
}

void main_nufft2(args::Subparser &parser)
{
  CoreArgs<2> coreArgs(parser);
  GridArgs<2> gridArgs(parser);
  PreconArgs  preArgs(parser);
  LSMRArgs    lsqOpts(parser);

  args::Flag fwd(parser, "", "Apply forward operator", {'f', "fwd"});
  args::Flag adj(parser, "", "Apply adjoint operator", {'a', "adj"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());

  TrajectoryN<2> traj(reader, reader.readInfo().voxel_size.head<2>(), coreArgs.matrix.Get());
  auto const     basis = LoadBasis(coreArgs.basisFile.Get());

  auto const shape = reader.dimensions();

  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(reader.readInfo());
  traj.write(writer);

  if (fwd) {
    auto const cart = reader.readTensor<Cx6>();
    auto const nS = shape[2];
    auto const nC = shape[3];
    auto const nT = shape[5];
    auto const nufft = TOps::NUFFT<2>::Make(gridArgs.Get(), traj, nC, basis.get());
    auto const A = Loopify2(nufft, nS, nT);
    auto const noncart = A->forward(cart);
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    auto const nC = shape[0];
    auto const nS = shape[3];
    auto const nT = shape[4];
    auto const nufft = TOps::NUFFT<2>::Make(gridArgs.Get(), traj, nC, basis.get());
    auto const A = Loopify2(nufft, nS, nT);
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
