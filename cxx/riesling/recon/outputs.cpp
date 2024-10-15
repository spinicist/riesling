#include "outputs.hpp"

#include "io/writer.hpp"
#include "log.hpp"

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

void WriteResidual(std::string const                &cmd,
                   std::string const                &fname,
                   Cx5                              &noncart,
                   Cx5Map const                     &x,
                   Info const                       &info,
                   typename TOps::TOp<Cx, 5, 5>::Ptr A,
                   Ops::Op<Cx>::Ptr                  M,
                  HD5::DimensionNames<5> const     &dims)
{
  Log::Print(cmd, "Calculating residual...");
  noncart -= A->forward(x);
  if (M) {
    Ops::Op<Cx>::Map  ncmap(noncart.data(), noncart.size());
    Ops::Op<Cx>::CMap nccmap(noncart.data(), noncart.size());
    M->inverse(nccmap, ncmap);
  }
  auto r = A->adjoint(noncart);
  Log::Print(cmd, "Finished calculating residual");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, r.dimensions(), r.data(), dims);
  Log::Print(cmd, "Wrote residual file {}", fname);
}

} // namespace rl