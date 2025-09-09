#include "inputs.hpp"

#include "rl/algo/otsu.hpp"
#include "rl/log/log.hpp"

#include <flux.hpp>

using namespace rl;

void main_mask(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Index>                           b(parser, "B", "Basis index", {'b', "basis"}, 0);
  args::ValueFlag<Index>                           t(parser, "T", "Time index", {'t', "time"}, 0);
  args::ValueFlag<float>                           thresh(parser, "T", "Use simple threshold instead of Otsu", {"thresh"});
  args::ValueFlag<Eigen::Vector3f, Vector3fReader> fov(parser, "FOV", "Crop to specified FOV (x,y,z)", {'f', "fov"});

  ParseCommand(parser, iname, oname);
  auto const cmd = parser.GetCommand().Name();
  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader ifile(iname.Get());
  auto const  info = ifile.readStruct<Info>(HD5::Keys::Info);
  Re3 const   in = ifile.readTensor<Cx5>().chip<4>(t.Get()).chip<3>(b.Get()).abs();
  float       thr = 0.f;
  if (thresh) {
    thr = thresh.Get();
  } else {
    auto const o = Otsu(CollapseToConstVector(in));
    thr = o.thresh;
  }
  Re3 out = (in > thr).select(in.constant(1.f), in.constant(0.f));

  if (fov) {
    Log::Print(cmd, "Masking to FOV {}", fov.Get());
    Eigen::Vector3f const                    ijk = (info.direction.inverse() * fov.Get()).array() / info.voxel_size;
    Sz3 const                                shape = out.dimensions();
    Sz3                                      mshape;
    Eigen::array<std::pair<Index, Index>, 3> paddings;
    for (Index ii = 0; ii < 3; ii++) {
      mshape[ii] = ijk[ii];
      if (mshape[ii] > shape[ii]) { throw Log::Failure(cmd, "Mask FOV exceeded image FOV"); }
      paddings[ii].first = (shape[ii] - mshape[ii]) / 2;
      paddings[ii].second = shape[ii] - mshape[ii] - paddings[ii].first;
    }
    Re3 m(mshape);
    m.setConstant(1.f);
    out.device(Threads::TensorDevice()) = out * m.pad(paddings);
  }

  HD5::Writer ofile(oname.Get());
  ofile.writeStruct(HD5::Keys::Info, info);
  ofile.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Image);
  Log::Print(cmd, "Finished");
}
