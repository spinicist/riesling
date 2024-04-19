#include "types.hpp"

#include "algo/lsmr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "threads.hpp"

using namespace rl;

void main_nufft(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());

  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  HD5::Writer writer(coreOpts.oname.Get());

  auto const start = Log::Now();
  Trajectory traj(reader, reader.readInfo().voxel_size);
  auto const nC = reader.dimensions()[0];
  auto       nufft = make_nufft(traj, gridOpts, nC, traj.matrixForFOV(coreOpts.fov.Get()), basis);

  if (fwd) {
    auto channels = reader.readTensor<Cx6>();
    Cx5  noncart(AddBack(nufft->oshape, channels.dimension(5)));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      noncart.chip<4>(ii).chip<3>(0).device(Threads::GlobalDevice()) = nufft->forward(CChipMap(channels, ii));
    }
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
    writer.writeInfo(reader.readInfo());
    traj.write(writer);
    Log::Print("Forward NUFFT took {}", Log::ToNow(start));
  } else {
    auto noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));

    auto const M = make_kspace_pre("kspace", nC, traj, basis);
    LSMR const lsmr{nufft, M, 4};

    Cx6 output(AddBack(nufft->ishape, noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(4); ii++) {
      output.chip<5>(ii).device(Threads::GlobalDevice()) = Tensorfy(lsmr.run(&noncart(0, 0, 0, 0, ii)), nufft->ishape);
    }
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Cartesian);
    Log::Print("NUFFT Adjoint took {}", Log::ToNow(start));
  }
}
