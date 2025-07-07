#include "inputs.hpp"

#include "rl/fft.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_resamp(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  ArrayFlag<float, 3>           res(parser, "R", "Target resolution (4 mm)", {"res"}, Eigen::Array3f::Constant(4.f));

  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader ifile(iname.Get());
  auto        input = ifile.readTensor<Cx5>();
  auto const  ishape = input.dimensions();
  auto const  inFo = ifile.readStruct<Info>(HD5::Keys::Info);

  Eigen::Array3f ratios = res.Get() / inFo.voxel_size;
  auto           oshape = ishape;
  auto           oFo = inFo;
  for (int ii = 0; ii < 3; ii++) {
    oshape[1 + ii] = ishape[1 + ii] / ratios[ii];
    oFo.voxel_size[ii] = inFo.voxel_size[ii] * ishape[1 + ii] / (1.f * oshape[1 + ii]);
  }

  FFT::Forward(input, Sz3{1, 2, 3});
  TOps::Pad<5> pad(ishape, oshape);
  Cx5          output(oshape);
  pad.forward(input, output);
  float const scale = std::sqrt(Product(oshape) / Product(ishape));
  Log::Print(cmd, "Scale {:3.2E}", scale);
  output.device(Threads::TensorDevice()) = output * Cx(scale);
  FFT::Adjoint(output, Sz3{1, 2, 3});

  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, oFo);
  writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Images);
  rl::Log::Print(cmd, "Finished");
}
