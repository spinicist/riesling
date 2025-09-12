#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include <flux.hpp>

using namespace rl;

template <int N> auto RemoveNth(HD5::DNames<N> const &in, Index const n) -> HD5::DNames<N - 1>
{
  HD5::DNames<N - 1> out;
  Index const        np1 = n + 1;
  std::copy_n(in.begin(), n, out.begin());
  std::copy_n(in.begin() + np1, N - np1, out.begin() + n);
  return out;
}

void main_rss(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output file name");
  args::ValueFlag<Index>        dim(parser, "DIM", "Dimension to take RSS (3)", {'d', "dim"}, 3);
  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  HD5::Writer writer(oname.Get());
  if (reader.exists(HD5::Keys::Info)) { writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info)); }
  auto const  order = reader.order();
  Index const D = dim.Get();
  switch (order) {
  case 4: {
    Cx4 const  in = reader.readTensor<Cx4>();
    auto const names = RemoveNth<4>(reader.readDNames<4>(), D);
    if (D < 0 || D > 3) { throw Log::Failure(cmd, "Dimension {} is invalid", D); }
    Cx3 const out = (in * in.conjugate()).sum(Sz1{D}).sqrt();
    writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), names);
  } break;
  case 5: {
    Cx5 const  in = reader.readTensor<Cx5>();
    auto const names = RemoveNth<5>(reader.readDNames<5>(), D);
    if (D < 0 || D > 4) { throw Log::Failure(cmd, "Dimension {} is invalid", D); }
    Cx4 const out = (in * in.conjugate()).sum(Sz1{D}).sqrt();
    writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), names);
  } break;
  case 6: {
    Cx6 const  in = reader.readTensor<Cx6>();
    auto const names = RemoveNth<6>(reader.readDNames<6>(), D);
    if (D < 0 || D > 5) { throw Log::Failure(cmd, "Dimension {} is invalid", D); }
    Cx5 const out = (in * in.conjugate()).sum(Sz1{D}).sqrt();
    writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), names);
  } break;
  default: throw Log::Failure(cmd, "Data had order {}", order);
  }
  Log::Print(cmd, "Finished");
}
