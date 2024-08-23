#include "outputs.hpp"

#include "io/writer.hpp"
#include "log.hpp"

namespace rl {

template <int ND>
void WriteOutput(
  std::string const &fname, CxN<ND> const &img, HD5::DimensionNames<ND> const &dims, Info const &info, std::string const &log)
{
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, img.dimensions(), img.data(), dims);
  if (log.size()) { writer.writeString("log", log); }
  Log::Print("Wrote output file {}", fname);
}

template void
WriteOutput<5>(std::string const &, Cx5 const &, HD5::DimensionNames<5> const &, Info const &, std::string const &);
template void
WriteOutput<6>(std::string const &, Cx6 const &, HD5::DimensionNames<6> const &, Info const &, std::string const &);

template <int ND>
void WriteResidual(std::string const                 &fname,
                   Cx5                               &noncart,
                   CxNCMap<ND> const                 &x,
                   Info const                        &info,
                   typename TOps::TOp<Cx, ND, 5>::Ptr A,
                   Ops::Op<Cx>::Ptr                   M,
                   HD5::DimensionNames<ND> const &dims)
{
  Log::Print("Calculating residual...");
  noncart -= A->forward(x);
  if (M) {
    Ops::Op<Cx>::Map  ncmap(noncart.data(), noncart.size());
    Ops::Op<Cx>::CMap nccmap(noncart.data(), noncart.size());
    M->inverse(nccmap, ncmap);
  }
  auto        r = A->adjoint(noncart);
  Log::Print("Finished calculating residual");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, r.dimensions(), r.data(), dims);
  Log::Print("Wrote residual file {}", fname);
}

template void WriteResidual<5>(std::string const &,
                               Cx5 &,
                               Cx5CMap const &,
                               Info const &,
                               TOps::TOp<Cx, 5, 5>::Ptr,
                               Ops::Op<Cx>::Ptr,
                               HD5::DimensionNames<5> const &);
template void WriteResidual<6>(std::string const &,
                               Cx5 &,
                               Cx6CMap const &,
                               Info const &,
                               TOps::TOp<Cx, 6, 5>::Ptr,
                               Ops::Op<Cx>::Ptr,
                               HD5::DimensionNames<6> const &);

} // namespace rl