#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

// Life is too short. Horrible template problems in Eigen.
template <int N, int M>
auto LastNames(HD5::Names<M> const &sz) -> HD5::Names<N>
{
  static_assert(N <= M);
  HD5::Names<N> last;
  std::copy_n(sz.end() - N, N, last.begin());
  return last;
}


int main_rss(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::ValueFlag<std::string>  oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset name (channels)", {'d', "dset"});
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "rss"));
  writer.writeInfo(reader.readInfo());

  auto const name = dset ? dset.Get() : HD5::Keys::Channels;
  auto const order = reader.order(name);

  switch (order) {
  case 5: {
    Cx5 const in = reader.readTensor<Cx5>(name);
    Cx4 const out = ConjugateSum(in, in).sqrt();
    writer.writeTensor(name, out.dimensions(), out.data(), LastNames<4>(reader.readDims<5>(name)));
  } break;
  case 6: {
    Cx6 const in = reader.readTensor<Cx6>(name);
    Cx5 const out = ConjugateSum(in, in).sqrt();
    writer.writeTensor(name, out.dimensions(), out.data(), LastNames<5>(reader.readDims<6>(name)));
  } break;
  default: Log::Fail("Dataset {} had order {}", name, order);
  }

  return EXIT_SUCCESS;
}
