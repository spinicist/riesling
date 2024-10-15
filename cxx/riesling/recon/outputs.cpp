#include "outputs.hpp"

#include "algo/lsmr.hpp"
#include "basis/basis.hpp"
#include "io/writer.hpp"
#include "log.hpp"
#include "op/recon.hpp"

namespace rl {

template <int ND>
void WriteOutput(
  std::string const &cmd, std::string const &fname, CxN<ND> const &img, HD5::DimensionNames<ND> const &dims, Info const &info)
{
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, img.dimensions(), img.data(), dims);
  if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
  Log::Print(cmd, "Wrote output file {}", fname);
}

template void
WriteOutput<5>(std::string const &, std::string const &, Cx5 const &, HD5::DimensionNames<5> const &, Info const &);
template void
WriteOutput<6>(std::string const &, std::string const &, Cx6 const &, HD5::DimensionNames<6> const &, Info const &);

void WriteResidual(std::string const              &cmd,
                   std::string const              &fname,
                   GridOpts                       &gridOpts,
                   SENSE::Opts                    &senseOpts,
                   PreconOpts                     &preOpts,
                   Trajectory const               &traj,
                   Cx5CMap const                  &x,
                   TOps::TOp<Cx, 5, 5>::Ptr const &A,
                   Cx5                            &noncart)
{
  Log::Print(cmd, "Creating recon operator without basis");
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);
  Basis const id;
  auto const  A1 = Recon::Choose(gridOpts, senseOpts, traj, &id, noncart);
  auto const  M1 = MakeKspacePre(traj, nC, nS, nT, &id, preOpts.type.Get(), preOpts.bias.Get());
  Log::Print(cmd, "Calculating K-space residual");
  noncart -= A->forward(x);
  Log::Print(cmd, "Calculating image residual");
  LSMR       lsmr{A1, M1, 2};
  auto const r = lsmr.run(CollapseToConstVector(noncart));
  Log::Print(cmd, "Finished calculating residual");
  HD5::Writer writer(fname, true);
  writer.writeTensor(HD5::Keys::Residual, A->ishape, r.data(), HD5::Dims::Image);
}

} // namespace rl