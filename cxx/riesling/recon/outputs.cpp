#include "outputs.hpp"

#include "io/writer.hpp"
#include "log.hpp"

namespace rl {

void WriteOutput(std::string const &fname, Cx5 const &img, Info const &info, std::string const &log)
{
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, img.dimensions(), img.data(), HD5::Dims::Image);
  if (log.size()) { writer.writeString("log", log); }
  Log::Print("Wrote output file {}", fname);
}

void WriteResidual(
  std::string const &fname, Cx5 &noncart, Cx5Map &x, Info const &info, TOps::TOp<Cx, 5, 5>::Ptr A, Ops::Op<Cx>::Ptr M)
{
  noncart -= A->forward(x);
  if (M) {
    Ops::Op<Cx>::Map  ncmap(noncart.data(), noncart.size());
    Ops::Op<Cx>::CMap nccmap(noncart.data(), noncart.size());
    M->inverse(nccmap, ncmap);
  }
  x = A->adjoint(noncart);
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, x.dimensions(), x.data(), HD5::Dims::Image);
  Log::Print("Wrote residual file {}", fname);
}

} // namespace rl