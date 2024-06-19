#include "basis/basis.hpp"
#include "colors.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "../magick.hpp"
#include "parse_args.hpp"
#include "tensors.hpp"
#include "types.hpp"

#include <scn/scan.h>

struct NameIndex
{
  std::string name;
  Index index;
};

struct NameIndexReader
{
  void operator()(std::string const &name, std::string const &value, NameIndex &ni)
  {
    if (auto result = scn::scan<std::string, Index>(value, "{:[a-z]},{:d}")) {
      ni.name = std::get<0>(result->values());
      ni.index = std::get<1>(result->values());
    } else {
      rl::Log::Fail("Could not read NameIndex for {} from value {}, error {}", name, value, result.error().msg());
    }
  }
};

struct GravityReader
{
  void operator()(std::string const &name, std::string const &value, Magick::GravityType &g)
  {
    if (value == "South") {
      g = Magick::GravityType::SouthGravity;
    } else if (value == "SouthWest") {
      g = Magick::GravityType::SouthWestGravity;
    } else if (value == "West") {
      g = Magick::GravityType::WestGravity;
    } else if (value == "NorthWest") {
      g = Magick::GravityType::NorthWestGravity;
    } else if (value == "North") {
      g = Magick::GravityType::NorthGravity;
    } else if (value == "NorthEast") {
      g = Magick::GravityType::NorthEastGravity;
    } else if (value == "East") {
      g = Magick::GravityType::EastGravity;
    } else if (value == "SouthEast") {
      g = Magick::GravityType::SouthEastGravity;
    } else {
      rl::Log::Fail("Unknown gravity {}", value);
    }
  }
};

auto ReadData(std::string const &iname, std::string const &dset, std::vector<NameIndex> namedChips) -> rl::Cx3
{
  rl::HD5::Reader reader(iname);
  auto const      diskOrder = reader.order(dset);
  auto const      diskDims = reader.dimensions(dset);

  rl::Cx3    data;
  auto const O = reader.order(dset);
  if (O < 2) {
    rl::Log::Fail("Cannot montage 1D data");
  } else if (O == 2) {
    auto const data2 = reader.readTensor<rl::Cx2>(dset);
    auto const shape = data2.dimensions();
    data = data2.reshape(rl::Sz3{shape[0], shape[1], 1});
  } else if (O == 3) {
    data = reader.readTensor<rl::Cx3>(dset);
  } else {
    std::vector<rl::HD5::IndexPair> chips;
    if (!namedChips.size()) {
      if (dset == "data") { chips = std::vector<rl::HD5::IndexPair>{{0, 0}, {4, 0}}; }
      else { rl::Log::Fail("No chips specified"); }
    } else {
      if (diskOrder - namedChips.size() != 3) {
        rl::Log::Fail("Incorrect chipping dimensions {}. Dataset {} has order {}, must be 3 after chipping", namedChips.size(), dset, diskOrder);
      }
      auto const names = reader.listNames(dset);
      for (auto const &nc: namedChips) {
        if (auto const id = std::find(names.cbegin(), names.cend(), nc.name); id != names.cend()) {
          auto const d = std::distance(names.cbegin(), id);
          if (nc.index >= diskDims[d]) {
            rl::Log::Fail("Dimension {} has {} slices asked for {}", nc.name, diskDims[d], nc.index);
          }
          chips.push_back({d, nc.index >= 0 ? nc.index : diskDims[d]  / 2});
        } else {
          rl::Log::Fail("Could find dimension named {}", nc.name);
        }
      }
    }
    data = reader.readSlab<rl::Cx3>(dset, chips);
  }
  rl::Log::Print("Data dims {}", data.dimensions());
  return data;
}

auto CrossSections(rl::Cx3 const &data) -> std::vector<rl::Cx2>
{
  auto const dShape = data.dimensions();

  std::vector<Magick::Image> slices(3);

  rl::Cx2 const X = data.chip<0>(dShape[0] / 2);
  rl::Cx2 const Y = data.chip<1>(dShape[1] / 2);
  rl::Cx2 const Z = data.chip<2>(dShape[2] / 2);
  rl::Log::Print("Cross section sizes {} {} {}", X.dimensions(), Y.dimensions(), Z.dimensions());
  return {X, Y, Z};
}

auto SliceData(rl::Cx3 const &data,
               Index const    slDim,
               Index const    slStart,
               Index const    slEnd,
               Index const    slN,
               rl::Sz2        sl0,
               rl::Sz2        sl1) -> std::vector<rl::Cx2>
{
  if (slDim < 0 || slDim > 2) { rl::Log::Fail("Slice dim was {}, must be 0-2", slDim); }
  auto const shape = data.dimensions();
  auto const shape1 = rl::Cx2(data.chip(0, slDim)).dimensions();
  if (slStart < 0 || slStart >= shape[slDim]) { rl::Log::Fail("Slice start invalid"); }
  if (slEnd && slEnd >= shape[slDim]) { rl::Log::Fail("Slice end invalid"); }
  if (slN < 0) { rl::Log::Fail("Requested negative number of slices"); }
  if (sl0[0] < 0) { sl0[0] = 0; }
  if (sl1[0] < 0) { sl1[0] = 0; }
  if (sl0[0] + sl0[1] >= shape1[0]) { sl0[1] = shape1[0] - sl0[0]; }
  if (sl1[0] + sl1[1] >= shape1[1]) { sl1[1] = shape1[1] - sl1[0]; }
  auto const start = slStart;
  auto const end = slEnd ? slEnd : shape[slDim] - 1;
  auto const maxN = end - start + 1;
  auto const N = slN ? std::min(slN, maxN) : maxN;
  auto const indices = Eigen::ArrayXf::LinSpaced(N, start, end);

  std::vector<rl::Cx2> slices;
  for (auto const index : indices) {
    rl::Cx2 temp = data.chip(std::floor(index), slDim).slice(rl::Sz2{sl0[0], sl1[0]}, rl::Sz2{sl0[1], sl1[1]});
    slices.push_back(temp);
  }
  rl::Log::Print("{} slices, dims {} {}", slices.size(), slices.front().dimension(0), slices.front().dimension(1));
  return slices;
}

auto Colorize(std::vector<rl::Cx2> const &slices, char const component, float const win, float const ɣ)
{
  std::vector<rl::RGBImage> colorized;
  for (auto const &slice : slices) {
    rl::RGBImage clr;
    switch (component) {
    case 'p': clr = rl::ColorizeComplex(slice / slice.abs().cast<rl::Cx>(), 1.f, 1.f); break;
    case 'x': clr = rl::ColorizeComplex(slice, win / 2.f, ɣ); break;
    case 'r': clr = rl::ColorizeReal(slice.real(), win, ɣ); break;
    case 'i': clr = rl::ColorizeReal(slice.imag(), win, ɣ); break;
    case 'm': clr = rl::Greyscale(slice.abs(), 0, win, ɣ); break;
    default: rl::Log::Fail("Uknown component type {}", component);
    }
    colorized.push_back(clr);
  }
  return colorized;
}

auto DoMontage(std::vector<rl::RGBImage> &slices, float const rotate, Index const colsIn) -> Magick::Image
{
  std::vector<Magick::Image> magicks(slices.size());
  std::transform(slices.begin(), slices.end(), magicks.begin(), [rotate](rl::RGBImage const &slc) {
    Magick::Image tmp(slc.dimension(1), slc.dimension(2), "RGB", Magick::CharPixel, slc.data());
    tmp.flip();
    tmp.rotate(rotate);
    return tmp;
  });

  auto const cols = slices.size() < colsIn ? slices.size() : colsIn;
  auto const rows = (Index)std::ceil(slices.size() / (float)colsIn);
  rl::Log::Print("Rows {} Cols {}", rows, cols);

  Magick::Montage montageOpts;
  montageOpts.backgroundColor(Magick::Color(0, 0, 0));
  montageOpts.tile(Magick::Geometry(cols, rows));
  montageOpts.geometry(magicks.front().size());
  std::vector<Magick::Image> frames;
  Magick::montageImages(&frames, magicks.begin(), magicks.end(), montageOpts);

  return frames.front();
}

void Colorbar(char const component, float const win, float const ɣ, Magick::Image &img)
{
  int const W = img.size().width() / 4;
  int const H = img.density().height() * img.fontPointsize() / 72.f;
  rl::Cx2   cx(W, H);
  for (Index ii = 0; ii < W; ii++) {
    float const mag = ii * win / (W - 1.f);
    for (Index ij = 0; ij < H; ij++) {
      switch (component) {
      case 'p': cx(ii, ij) = std::polar<float>(1.f, -M_PI + 2.f * M_PI * ii / (W - 1.f)); break;
      case 'x': cx(ii, ij) = std::polar<float>(mag, -M_PI + 2.f * M_PI * ij / (H - 1.f)); break;
      case 'r': cx(ii, ij) = 2.f * mag - win; break;
      case 'i': cx(ii, ij) = 2.f * mag - win; break;
      case 'm': cx(ii, ij) = mag; break;
      default: rl::Log::Fail("Uknown component type {}", component);
      }
    }
  }
  rl::RGBImage cbar;
  std::string  leftLabel, rightLabel;
  switch (component) {
  case 'p': {
    leftLabel = "-ᴨ";
    rightLabel = "ᴨ";
    cbar = rl::ColorizeComplex(cx, 1.f, 1.f);
  } break;
  case 'x': {
    leftLabel = "0";
    rightLabel = fmt::format("{:.1f}", win);
    cbar = rl::ColorizeComplex(cx, win / 2.f, ɣ);
  } break;
  case 'r': {
    leftLabel = fmt::format("{:.1f}", -win);
    rightLabel = fmt::format("{:.1f}", win);
    cbar = rl::ColorizeReal(cx.real(), win, ɣ);
  } break;
  case 'i': {
    leftLabel = fmt::format("{:.1f}", -win);
    rightLabel = fmt::format("{:.1f}", win);
    cbar = rl::ColorizeReal(cx.real(), win, ɣ);
  } break;
  case 'm': {
    leftLabel = "0";
    rightLabel = fmt::format("{:.1f}", win);
    cbar = rl::Greyscale(cx.abs(), 0, win, ɣ);
  } break;
  default: rl::Log::Fail("Uknown component type {}", component);
  }

  Magick::Image          cbarImg(W, H, "RGB", Magick::CharPixel, cbar.data());
  Magick::Geometry const cbarTextBounds(W, H, W * 0.01);

  cbarImg.density(img.density());
  cbarImg.font(img.font());
  cbarImg.fontPointsize(img.fontPointsize());
  cbarImg.fillColor(Magick::ColorMono(true));
  cbarImg.strokeWidth(4);
  cbarImg.strokeColor(Magick::ColorMono(false));
  cbarImg.annotate(leftLabel, cbarTextBounds, Magick::WestGravity);
  cbarImg.annotate(rightLabel, cbarTextBounds, Magick::EastGravity);
  cbarImg.strokeColor(Magick::Color("none"));
  cbarImg.annotate(leftLabel, cbarTextBounds, Magick::WestGravity);
  cbarImg.annotate(rightLabel, cbarTextBounds, Magick::EastGravity);

  cbarImg.borderColor(Magick::ColorGray(0.));
  cbarImg.border(Magick::Geometry(4, 4));

  img.composite(cbarImg, Magick::SouthGravity);
}

void Decorate(std::string const &title, Magick::GravityType const gravity, Magick::Image &montage)
{
  montage.fillColor(Magick::ColorMono(true));
  montage.strokeColor(Magick::ColorMono(false));
  montage.strokeWidth(4);
  montage.annotate(title, gravity);
  montage.strokeColor(Magick::Color("none"));
  montage.annotate(title, gravity);
}

void Printify(std::string const &oname, float const printPixWidth, bool const interp, Magick::Image &img)
{
  float const scale = printPixWidth / (float)img.size().width();
  if (interp) {
    img.resize(Magick::Geometry(printPixWidth, scale * img.size().height()));
  } else {
    img.scale(Magick::Geometry(printPixWidth, scale * img.size().height()));
  }
  img.density(Magick::Geometry(72, 72));
}

void main_montage(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to slice");
  args::Positional<std::string> oname(parser, "FILE", "Image file to save");
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset (image)", {"dset", 'd'}, "data");

  args::ValueFlagList<NameIndex, std::vector, NameIndexReader> chips(parser, "C", "Chip a dimension", {"chip", 'c'});

  args::ValueFlag<std::string>                        title(parser, "T", "Title", {"title", 't'});
  args::ValueFlag<Magick::GravityType, GravityReader> gravity(parser, "G", "Title gravity", {"gravity"}, Magick::NorthGravity);
  args::ValueFlag<std::string>                        font(parser, "F", "Font", {"font", 'f'}, "Helvetica");
  args::ValueFlag<float>                              fontSize(parser, "FS", "Font size", {"font-size"}, 18);

  args::ValueFlag<int> width(parser, "W", "Width in pixels for figures", {"width", 'w'}, 1800);
  args::Flag           interp(parser, "I", "Use proper interpolation", {"interp"});

  args::Flag noScale(parser, "N", "Don't scale to terminal width", {"no-scale"});

  args::ValueFlag<char>  comp(parser, "C", "Component (x,r,i,m,p)", {"comp"}, 'x');
  args::ValueFlag<float> max(parser, "W", "Max intensity", {"max"});
  args::ValueFlag<float> maxP(parser, "P", "Max intensity as %", {"maxP"}, 0.9);
  args::ValueFlag<float> ɣ(parser, "G", "Gamma correction", {"gamma"}, 1.f);
  args::Flag             cbar(parser, "C", "Add colorbar", {"cbar"});

  args::ValueFlag<Index>                slN(parser, "N", "Number of slices (0 for all)", {"num", 'n'}, 8);
  args::ValueFlag<Index>                slStart(parser, "S", "Start slice", {"start"}, 0);
  args::ValueFlag<Index>                slEnd(parser, "S", "End slice", {"end"});
  args::ValueFlag<Index>                slDim(parser, "S", "Slice dimension (0/1/2)", {"dim"}, 0);
  args::ValueFlag<rl::Sz2, SzReader<2>> sl0(parser, "S", "Dim 0 slice (start, size)", {"sl0"}, rl::Sz2{0, 1024});
  args::ValueFlag<rl::Sz2, SzReader<2>> sl1(parser, "S", "Dim 1 slice (start, size)", {"sl1"}, rl::Sz2{0, 1024});
  args::ValueFlag<Index>                cols(parser, "C", "Output columns", {"cols"}, 8);
  args::Flag                            cross(parser, "C", "Cross sections", {"cross-sections", 'x'});
  args::ValueFlag<float>                rotate(parser, "D", "Rotate slices (degrees)", {"rot", 'r'}, 0.f);
  ParseCommand(parser, iname);
  Magick::InitializeMagick(NULL);

  auto const  data = ReadData(iname.Get(), dset.Get(), chips.Get());
  float const maxData = rl::Maximum(data.abs());
  float const winMax = max ? max.Get() : maxP.Get() * maxData;
  rl::Log::Print("Max magnitude in data {}. Window maximum {}", maxData, winMax);

  auto slices =
    cross ? CrossSections(data) : SliceData(data, slDim.Get(), slStart.Get(), slEnd.Get(), slN.Get(), sl0.Get(), sl1.Get());
  auto colorized = Colorize(slices, comp.Get(), winMax, ɣ.Get());
  auto montage = DoMontage(colorized, rotate.Get(), cols.Get());
  rl::Log::Print("Image size: {} {}", montage.size().width(), montage.size().height());
  if (oname) { Printify(oname.Get(), width.Get(), interp, montage); }
  montage.font(font.Get());
  montage.fontPointsize(fontSize.Get());
  if (cbar) { Colorbar(comp.Get(), winMax, ɣ.Get(), montage); }
  Decorate(title ? title.Get() : fmt::format("{} {}", iname.Get(), dset.Get()), gravity.Get(), montage);
  if (oname) {
    montage.magick("PNG");
    montage.write(oname.Get());
  } else {
    rl::ToKitty(montage, !noScale);
  }
}
