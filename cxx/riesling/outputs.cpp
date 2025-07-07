#include "outputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/basis/basis.hpp"
#include "rl/io/writer.hpp"
#include "rl/log/log.hpp"
#include "rl/op/recon.hpp"

namespace rl {

template <size_t ND> void
WriteOutput(std::string const &cmd, std::string const &fname, CxN<ND> const &img, HD5::DNames<ND> const &dims, Info const &info)
{
  HD5::Writer writer(fname);
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, img.dimensions(), img.data(), dims);
  if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
  Log::Print(cmd, "Wrote output file {}", fname);
}

template void WriteOutput<5>(std::string const &, std::string const &, Cx5 const &, HD5::DNames<5> const &, Info const &);
template void WriteOutput<6>(std::string const &, std::string const &, Cx6 const &, HD5::DNames<6> const &, Info const &);

void WriteResidual(std::string const              &cmd,
                   std::string const              &fname,
                   ReconOpts const                &reconOpts,
                   GridOpts<3> const              &gridOpts,
                   SENSE::Opts<3> const           &senseOpts,
                   PreconOpts const               &preOpts,
                   Trajectory const               &traj,
                   Cx5CMap                  &x,
                   TOps::TOp<5, 5>::Ptr const &A,
                   Cx5                            &noncart)
{
  Log::Print(cmd, "Creating recon operator without basis");
  Basis const id;
  auto const  R1 = Recon(reconOpts, preOpts, gridOpts, senseOpts, traj, &id, noncart);
  Log::Print(cmd, "Calculating K-space residual");
  noncart.device(Threads::TensorDevice()) -= A->forward(x);
  Log::Print(cmd, "Calculating image residual");
  LSMR       lsmr{R1.A, R1.M, nullptr, {2}};
  auto const r = lsmr.run(CollapseToConstVector(noncart));
  Log::Print(cmd, "Finished calculating residual");
  HD5::Writer writer(fname, true);
  writer.writeTensor(HD5::Keys::Residual, R1.A->ishape, r.data(), HD5::Dims::Images);
}

} // namespace rl