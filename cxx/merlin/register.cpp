#include "merlin.hpp"

#include "../args.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"

#include "util.hpp"

void main_reg(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "ifile HD5 file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output HD5 file");
  SzFlag<6>                     mask(parser, "MASK", "Mask HD5 file", {'m', "mask"});
  args::ValueFlag<Index>        nav(parser, "NAV", "Only register one navigator", {'n', "nav"});
  ParseCommand(parser, iname, oname);
  auto const cmd = parser.GetCommand().Name();

  merlin::ImageType::RegionType maskRegion;
  if (mask) {
    auto const                               m = mask.Get();
    merlin::ImageType::RegionType::SizeType  sz;
    merlin::ImageType::RegionType::IndexType ind;
    for (Index ii = 0; ii < 3; ii++) {
      ind[ii] = m[ii * 2];
      sz[ii] = m[ii * 2 + 1];
    }
    maskRegion.SetSize(sz);
    maskRegion.SetIndex(ind);
  }

  rl::HD5::Reader ifile(iname.Get());
  rl::HD5::Writer ofile(oname.Get());

  rl::Re4    idata = ifile.readTensor<rl::Cx5>().abs().chip<4>(0); // ITK does not like this being const
  auto const info = ifile.readStruct<rl::Info>(rl::HD5::Keys::Info);
  auto const fixed = merlin::Import(rl::ChipMap(idata, 0), info);

  merlin::MERLIN wizard(fixed, maskRegion);
  merlin::TransformType::Pointer tfm = nullptr;
  if (nav) {
    Index inav = nav.Get();
    if (inav < 0 || inav >= idata.dimension(3)) {
      throw rl::Log::Failure(cmd, "Specified navigator {} outside valid range", inav);
    }
    auto const moving = merlin::Import(rl::ChipMap(idata, inav), info);
    rl::Log::Print("MERLIN", "Register navigator {} to {}", inav, 0);
    tfm = wizard.registerMoving(moving, tfm);
    ofile.writeStruct(fmt::format("{:02d}", inav), merlin::ITKToRIESLING(tfm));
  } else {
    for (Index ii = 1; ii < idata.dimension(3); ii++) {
      auto const moving = merlin::Import(rl::ChipMap(idata, ii), info);
      rl::Log::Print("MERLIN", "Register navigator {} to {}", ii, 0);
      tfm = wizard.registerMoving(moving, tfm);
      ofile.writeStruct(fmt::format("{:04d}", ii), merlin::ITKToRIESLING(tfm));
    }
  }
  rl::Log::Print("MERLIN", "Finished");
}