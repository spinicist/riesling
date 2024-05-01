#include "basis/basis.hpp"
#include "colors.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "types.hpp"

#include "Magick++.h"
#include <Eigen/Core>
#include <ranges>
#include <scn/scan.h>
#include <sys/ioctl.h>
#include <tl/chunk.hpp>
#include <tl/to.hpp>

struct IndexPairReader
{
  void operator()(std::string const &name, std::string const &value, IndexPair &p)
  {
    if (auto result = scn::scan<Index, Index>(value, "{},{}")) {
      p.dim = std::get<0>(result->values());
      p.index = std::get<1>(result->values());
    } else {
      rl::Log::Fail("Could not read Index Pair for {} from value {}", name, value);
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

auto ReadData(std::string const &iname, std::string const &dset, std::vector<IndexPair> chips, char const component) -> rl::Cx3
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
  } else {
    if (!chips.size()) {
      if (dset == "data") { chips = std::vector<IndexPair>{{0, 0}, {4, 0}}; }
    } else {
      if (diskOrder - chips.size() != 3) {
        rl::Log::Fail("Dataset {} has order {} and only {} chips", dset, diskOrder, chips.size());
      }
    }
    data = reader.readSlab<rl::Cx3>(dset, chips);
  }
  switch (component) {
  case 'x': break;
  case 'r': data = data.real().cast<rl::Cx>(); break;
  case 'i': data = data.imag().cast<rl::Cx>(); break;
  case 'm': data = data.abs().cast<rl::Cx>(); break;
  case 'p': data = data.arg().cast<rl::Cx>(); break;
  default: rl::Log::Fail("Uknown component type {}", component);
  }
  rl::Log::Print("Data dims {}", data.dimensions());
  return data;
}

auto CrossSections(rl::Cx3 const &data, float const win, bool const grey, float const ɣ, float const rotate)
  -> std::vector<Magick::Image>
{
  auto const dShape = data.dimensions();

  std::vector<Magick::Image> slices(3);

  rl::Cx2 const X = data.chip<0>(dShape[0] / 2);
  rl::Cx2 const Y = data.chip<1>(dShape[1] / 2);
  rl::Cx2 const Z = data.chip<2>(dShape[2] / 2);
  auto const    sliceX = rl::Colorize(X, win, grey, ɣ);
  auto const    sliceY = rl::Colorize(Y, win, grey, ɣ);
  auto const    sliceZ = rl::Colorize(Z, win, grey, ɣ);
  rl::Log::Print("Cross section sizes {} {} {}", sliceX.dimensions(), sliceY.dimensions(), sliceZ.dimensions());
  slices[0] = Magick::Image(sliceX.dimension(1), sliceX.dimension(2), "RGB", Magick::CharPixel, sliceX.data());
  slices[1] = Magick::Image(sliceY.dimension(1), sliceY.dimension(2), "RGB", Magick::CharPixel, sliceY.data());
  slices[2] = Magick::Image(sliceZ.dimension(1), sliceZ.dimension(2), "RGB", Magick::CharPixel, sliceZ.data());
  slices[0].flip(); // Reverse Y for display
  slices[0].rotate(rotate);
  slices[1].flip(); // Reverse Y for display
  slices[1].rotate(rotate);
  slices[2].flip(); // Reverse Y for display
  slices[2].rotate(rotate);
  return slices;
}

auto SliceData(rl::Cx3 const &data,
               Index const    slDim,
               Index const    slStart,
               Index const    slEnd,
               Index const    slN,
               rl::Sz2        sl0,
               rl::Sz2        sl1,
               float const    win,
               bool const     grey,
               float const    ɣ,
               float const    rotate) -> std::vector<Magick::Image>
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

  std::vector<Magick::Image> slices;
  for (auto const index : indices) {
    rl::Cx2    temp = data.chip(std::floor(index), slDim).slice(rl::Sz2{sl0[0], sl1[0]}, rl::Sz2{sl0[1], sl1[1]});
    auto const slice = rl::Colorize(temp, win, grey, ɣ);
    slices.push_back(Magick::Image(sl0[1], sl1[1], "RGB", Magick::CharPixel, slice.data()));
    slices.back().flip(); // Reverse Y for display
    slices.back().rotate(rotate);
  }
  rl::Log::Print("{} slices, dims {} {}", slices.size(), slices.front().size().height(), slices.front().size().width());
  return slices;
}

auto DoMontage(std::vector<Magick::Image> &slices, Index const colsIn) -> Magick::Image
{
  auto const cols = slices.size() < colsIn ? slices.size() : colsIn;
  auto const rows = (Index)std::ceil(slices.size() / (float)colsIn);
  rl::Log::Print("Rows {} Cols {}", rows, cols);

  Magick::Montage montageOpts;
  montageOpts.backgroundColor(Magick::Color(0, 0, 0));
  montageOpts.tile(Magick::Geometry(cols, rows));
  montageOpts.geometry(slices.front().size());
  std::vector<Magick::Image> frames;
  Magick::montageImages(&frames, slices.begin(), slices.end(), montageOpts);

  return frames.front();
}

void Resize(bool const print, float const printPixWidth, bool const interp, Magick::Image &img)
{
  if (print) {
    float const scale = printPixWidth / (float)img.size().width();
    if (interp) {
      img.resize(Magick::Geometry(printPixWidth, scale * img.size().height()));
    } else {
      img.scale(Magick::Geometry(printPixWidth, scale * img.size().height()));
    }
    img.density(Magick::Geometry(72, 72));
  } else {
    struct winsize winSize;
    ioctl(0, TIOCGWINSZ, &winSize);
    auto const scaling = winSize.ws_xpixel / (float)img.size().width();
    if (interp) {
      img.resize(Magick::Geometry(winSize.ws_xpixel, scaling * img.size().height()));
    } else {
      img.scale(Magick::Geometry(winSize.ws_xpixel, scaling * img.size().height()));
    }
    img.density(Magick::Geometry(90, 90));
  }
}

void Colorbar(float const win, bool const grey, float const ɣ, Magick::Image &img)
{
  int const W = img.size().width() / 4;
  int const H = img.density().height() * img.fontPointsize() / 72.f;
  rl::Cx2   cx(W, H);
  for (Index ii = 0; ii < W; ii++) {
    // if (grey) {
      float const mag = ii * win / (W - 1.f);
      cx.chip<0>(ii).setConstant(mag);
    // } else {
    //   float const angle = -M_PI + 2.f * M_PI * ii / (W - 1.f);
    //   cx.chip<0>(ii).setConstant(std::polar(0.5f, angle));
    // }
  }
  auto const             cbar = rl::Colorize(cx, win, grey, ɣ);
  Magick::Image          cbarImg(W, H, "RGB", Magick::CharPixel, cbar.data());
  Magick::Geometry const cbarTextBounds(W, H, W * 0.01);

  std::string const leftLabel = "0";
  std::string const rightLabel = fmt::format("{:.2f}", win);

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

void Kittify(Magick::Image &graphic)
{
  Magick::Blob blob;
  graphic.write(&blob);
  auto const     b64 = blob.base64();
  constexpr auto ChunkSize = 4096;
  if (b64.size() <= ChunkSize) {
    fmt::print(stderr, "\x1B_Ga=T,f=100;{}\x1B\\", b64);
  } else {
    auto const chunks = b64 | tl::views::chunk(ChunkSize);
    auto const nChunks = chunks.size();
    fmt::print(stderr, "\x1B_Ga=T,f=100,m=1;{}\x1B\\", std::string_view(chunks.front().data(), chunks.front().size()));
    for (auto &&chunk : chunks | std::ranges::views::drop(1) | std::ranges::views::take(nChunks - 2)) {
      fmt::print(stderr, "\x1B_Gm=1;{}\x1B\\", std::string_view(chunk.data(), chunk.size()));
    }
    fmt::print(stderr, "\x1B_Gm=0;{}\x1B\\", std::string_view(chunks.back().data(), chunks.back().size()));
  }
  fmt::print(stderr, "\n");
}

void main_montage(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to slice");
  args::Positional<std::string> oname(parser, "FILE", "Image file to save");
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset (image)", {"dset", 'd'}, "data");

  args::ValueFlagList<IndexPair, std::vector, IndexPairReader> chips(parser, "C", "Chip a dimension", {"chip", 'c'});

  args::ValueFlag<std::string>                        title(parser, "T", "Title", {"title", 't'});
  args::ValueFlag<Magick::GravityType, GravityReader> gravity(parser, "G", "Title gravity", {"gravity"}, Magick::NorthGravity);
  args::ValueFlag<std::string>                        font(parser, "F", "Font", {"font", 'f'}, "Helvetica");
  args::ValueFlag<float>                              fontSize(parser, "FS", "Font size", {"font-size"}, 18);

  args::ValueFlag<Index> cols(parser, "C", "Output columns", {"cols"}, 8);
  args::ValueFlag<int>   width(parser, "W", "Width in pixels for figures", {"width", 'w'}, 1800);
  args::Flag             interp(parser, "I", "Use proper interpolation", {"interp"});

  args::ValueFlag<char>  comp(parser, "C", "Component (x,r,i,m,p)", {"comp"}, 'x');
  args::ValueFlag<float> max(parser, "W", "Max intensity", {"max"});
  args::ValueFlag<float> maxP(parser, "P", "Max intensity as %", {"maxP"}, 0.9);
  args::Flag             grey(parser, "G", "Greyscale", {"grey", 'g'});
  args::ValueFlag<float> ɣ(parser, "G", "Gamma correction", {"gamma"}, 1.f);
  args::Flag             cbar(parser, "C", "Add colorbar", {"cbar"});

  args::ValueFlag<Index>                slN(parser, "N", "Number of slices (0 for all)", {"num", 'n'}, 8);
  args::ValueFlag<Index>                slStart(parser, "S", "Start slice", {"start"}, 0);
  args::ValueFlag<Index>                slEnd(parser, "S", "End slice", {"end"});
  args::ValueFlag<Index>                slDim(parser, "S", "Slice dimension (0/1/2)", {"dim"}, 0);
  args::ValueFlag<rl::Sz2, SzReader<2>> sl0(parser, "S", "Dim 0 slice (start, size)", {"sl0"}, rl::Sz2{0, 1024});
  args::ValueFlag<rl::Sz2, SzReader<2>> sl1(parser, "S", "Dim 1 slice (start, size)", {"sl1"}, rl::Sz2{0, 1024});
  args::Flag                            cross(parser, "C", "Cross sections", {"cross-sections", 'x'});
  args::ValueFlag<float>                rotate(parser, "D", "Rotate slices (degrees)", {"rot", 'r'}, 0.f);
  ParseCommand(parser, iname);
  Magick::InitializeMagick(NULL);

  auto const  data = ReadData(iname.Get(), dset.Get(), chips.Get(), comp.Get());
  float const maxData = rl::Maximum(data.abs());
  float const winMax = max ? max.Get() : maxP.Get() * maxData;
  rl::Log::Print("Max magnitude in data {}. Window maximum {}", maxData, winMax);

  auto slices = cross ? CrossSections(data, winMax, grey, ɣ.Get(), rotate.Get())
                      : SliceData(data, slDim.Get(), slStart.Get(), slEnd.Get(), slN.Get(), sl0.Get(), sl1.Get(), winMax, grey,
                                  ɣ.Get(), rotate.Get());
  auto montage = DoMontage(slices, cols.Get());
  rl::Log::Print("Initial image size: {} {}", montage.size().width(), montage.size().height());
  Resize(oname, width.Get(), interp, montage);
  rl::Log::Print("Final image size: {} {}", montage.size().width(), montage.size().height());
  montage.font(font.Get());
  montage.fontPointsize(fontSize.Get());
  if (cbar) { Colorbar(winMax, grey, ɣ.Get(), montage); }
  Decorate(title ? title.Get() : fmt::format("{} {}", iname.Get(), dset.Get()), gravity.Get(), montage);
  montage.magick("PNG");
  if (oname) {
    montage.write(oname.Get());
  } else {
    Kittify(montage);
  }
}
