#include "basis/basis.hpp"
#include "colors.hpp"
#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "types.hpp"

#include <notcurses/notcurses.h>

using namespace rl;

// Loads of global variables because notcurses callbacks need function pointers and capturing Lambdas can't be function pointers
notcurses  *nc = nullptr;
ncplane    *ncstd = nullptr, *pX = nullptr, *pY = nullptr, *pZ = nullptr, *pLog = nullptr;
std::string status;

void BlitImage(RGBImage const &img, ncplane *ncp)
{
  auto vopts = ncvisual_options{};
  vopts.flags = NCVISUAL_OPTION_NOINTERPOLATE;
  vopts.scaling = NCSCALE_SCALE;
  vopts.blitter = NCBLIT_PIXEL;
  vopts.n = ncp;
  auto v = ncvisual_from_rgb_packed(img.data(), img.dimension(2), img.dimension(1) * 3, img.dimension(1), 255);
  ncvisual_blit(nc, v, &vopts);
  ncvisual_destroy(v);
}

auto GetInput() -> char32_t
{
  char32_t input = 0;
  ncinput  ni;
  timespec const ts{.tv_sec = 0, .tv_nsec = 50000};
  while (true) {
    input = notcurses_get(nc, &ts, &ni);
    if (input == (char32_t)-1) { continue; }
    if (ni.evtype == NCTYPE_RELEASE) { continue; }
    if (ni.evtype == NCTYPE_UNKNOWN) { continue; }
    break;
  }
  return input;
}

int Resize(ncplane *ncp);
void InitUI()
{
  notcurses_options nopts{};
  nopts.flags |= NCOPTION_NO_CLEAR_BITMAPS | NCOPTION_INHIBIT_SETLOCALE;
  if ((nc = notcurses_init(&nopts, nullptr)) == nullptr) { Log::Fail("Could not initialize notcurses"); }

  ncstd = notcurses_stdplane(nc);
  ncplane_options opts{};
  opts.rows = 1;
  opts.cols = 1;
  if (!(pX = ncplane_create(ncstd, &opts))) { Log::Fail("Could not create plane"); };
  if (!(pY = ncplane_create(ncstd, &opts))) { Log::Fail("Could not create plane"); };
  if (!(pZ = ncplane_create(ncstd, &opts))) { Log::Fail("Could not create plane"); };
  if (!(pLog = ncplane_create(ncstd, &opts))) { Log::Fail("Could not create plane"); };
  Resize(ncstd);
  ncplane_set_resizecb(ncstd, Resize);
}

int Resize(ncplane *ncp)
{
  unsigned pixY, pixX, cellpixY, cellpixX, maxpixX, maxpixY;
  ncplane_pixel_geom(ncstd, &pixY, &pixX, &cellpixY, &cellpixX, &maxpixY, &maxpixX);
  auto const rows = pixY / cellpixY;
  auto const cols = pixX / cellpixX;

  ncplane_resize_simple(pX, rows / 2, cols / 3);
  ncplane_move_yx(pX, 1, 0);
  ncplane_resize_simple(pY, rows / 2, cols / 3);
  ncplane_move_yx(pY, 1, cols / 3);
  ncplane_resize_simple(pZ, rows / 2, cols / 3);
  ncplane_move_yx(pZ, 1, cols * 2 / 3);
  ncplane_resize_simple(pLog, rows / 2, cols);
  ncplane_move_yx(pLog, rows / 2, 0);
  return true;
}

void BlitXYZ(RGBImage const &imgX, RGBImage const &imgY, RGBImage const &imgZ)
{
  BlitImage(imgX, pX);
  BlitImage(imgY, pY);
  BlitImage(imgZ, pZ);
}

void Render()
{
  ncplane_cursor_move_yx(ncstd, 0, 0);
  ncplane_printf(ncstd, "%s", status.c_str());
  notcurses_render(nc);
}

int main_view(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to view");
  ParseCommand(parser);

  InitUI();

  HD5::Reader reader(iname.Get());
  Cx5         img = reader.readTensor<Cx5>("image");

  Index const nV = img.dimension(0);
  Index const nI = img.dimension(1);
  Index const nJ = img.dimension(2);
  Index const nK = img.dimension(3);
  Index const nT = img.dimension(4);
  Index       iV = nV / 2;
  Index       iI = nI / 2;
  Index       iJ = nJ / 2;
  Index       iK = nK / 2;
  Index       iT = nT / 2;
  float       magLim = 0.9 * Maximum(img.abs());
  bool        grey = true;

  bool  go = true;
  Index inc = 1;
  while (go) {
    Cx2 sliceX = img.chip<4>(iT).chip<1>(iI).chip<0>(iV);
    Cx2 sliceY = img.chip<4>(iT).chip<2>(iJ).chip<0>(iV);
    Cx2 sliceZ = img.chip<4>(iT).chip<3>(iK).chip<0>(iV);

    auto const imgX = Colorize(sliceX, magLim, grey);
    auto const imgY = Colorize(sliceY, magLim, grey);
    auto const imgZ = Colorize(sliceZ, magLim, grey);

    BlitXYZ(imgX, imgY, imgZ);
    status = fmt::format("{} B {}/{} I {}/{} J {}/{} K {}/{} V {}/{} Mag Lim {}", iname.Get(), iV, nV - 1, iI, nI - 1, iJ,
                            nJ - 1, iK, nK - 1, iV, nV - 1, magLim);
    Render();

    auto const input = GetInput();
    if (input == 'x') { break; }
    switch (input) {
    case 'q': iV = std::max(iV - inc, 0L); break;
    case 'a': iV = std::min(iV + inc, nV - 1); break;
    case 'w': iI = std::max(iI - inc, 0L); break;
    case 's': iI = std::min(iI + inc, nI - 1); break;
    case 'e': iJ = std::max(iJ - inc, 0L); break;
    case 'd': iJ = std::min(iJ + inc, nJ - 1); break;
    case 'r': iK = std::max(iK - inc, 0L); break;
    case 'f': iK = std::min(iK + inc, nK - 1); break;
    case 't': iT = std::max(iT - inc, 0L); break;
    case 'g': iT = std::min(iT + inc, nT - 1); break;
    case 'p': magLim = magLim * 0.9; break;
    case 'l': magLim = magLim * 1.1; break;
    case 'o': grey = !grey; break;
    case '1': inc = 1; break;
    case '2': inc = 2; break;
    case '3': inc = 3; break;
    case '4': inc = 4; break;
    case '5': inc = 5; break;
    case '6': inc = 6; break;
    case '7': inc = 7; break;
    case '8': inc = 8; break;
    case '9': inc = 9; break;
    case '0': inc = 10; break;
    default: break;
    }
  }

  return notcurses_stop(nc);
}
