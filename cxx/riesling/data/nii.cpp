#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/io/nifti.hpp"
#include "rl/log.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_nii(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output nii file");
  args::Flag                    real(parser, "REAL", "Write magnitude images", {"real", 'r'});
  args::Flag                    mag(parser, "MAGNITUDE", "Write magnitude images", {"mag", 'm'});
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset name (default image)", {'d', "dset"}, "data");

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader input(iname.Get());
  Info const  info = input.readInfo();

  Cx4 output;
  if (input.order() == 3) {
    Cx3 const image = input.readTensor<Cx3>(dset.Get());
    Sz3 const sz = image.dimensions();
    output = image.reshape(Sz4{sz[0], sz[1], sz[2], 1});
  } else if (input.order() == 4) {
    output = input.readTensor<Cx4>(dset.Get());
  } else if (input.order() == 5) {
    Cx5 const image = input.readTensor<Cx5>(dset.Get());
    Sz5 const sz = image.dimensions();
    output = image.reshape(Sz4{sz[0], sz[1], sz[2], sz[3] * sz[4]});
  } else if (input.order() == 6) {
    Cx6 const image = input.readTensor<Cx6>(dset.Get());
    Sz6 const sz = image.dimensions();
    output = image.reshape(Sz4{sz[0], sz[1], sz[2], sz[3] * sz[4] * sz[5]});
  } else {
    throw Log::Failure(cmd, "Dataset {} was order {}, needs to be 5 or 6", dset.Get(), input.order());
  }

  if (mag) {
    WriteNifti(info, Re4(output.abs()), oname.Get());
  } else if (real) {
    WriteNifti(info, Re4(output.real()), oname.Get());
  } else {
    WriteNifti(info, output, oname.Get());
  }
  Log::Print(cmd, "Finished");
}
