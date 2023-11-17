#include "basis/basis.hpp"
#include "img/rgb.hpp"
#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "types.hpp"

#include <notcurses/notcurses.h>

using namespace rl;

struct UI
{
  notcurses *nc = nullptr;
  ncplane   *pStd, *pX, *pY, *pZ;
  unsigned   rows, cols;
  UI()
  {
    notcurses_options nopts{};
    nopts.flags |= NCOPTION_NO_CLEAR_BITMAPS | NCOPTION_INHIBIT_SETLOCALE;
    if ((nc = notcurses_init(&nopts, nullptr)) == nullptr) { Log::Fail("Could not initialize notcurses"); }
    pStd = notcurses_stdplane(nc);
    updateGeom();

    ncplane_options opts{};
    opts.x = 0;
    opts.y = 1;
    opts.cols = cols / 3;
    opts.rows = rows - 1;
    pX = ncplane_create(pStd, &opts);
    opts.x = cols / 3;
    pY = ncplane_create(pStd, &opts);
    opts.x = cols * 2 / 3;
    pZ = ncplane_create(pStd, &opts);
  }

  auto stop() -> bool { return notcurses_stop(nc); }

  void updateGeom()
  {
    unsigned pixY, pixX, cellpixY, cellpixX, maxpixX, maxpixY;
    ncplane_pixel_geom(pStd, &pixY, &pixX, &cellpixY, &cellpixX, &maxpixY, &maxpixX);
    rows = pixY / cellpixY;
    cols = pixX / cellpixX;
  }

  void blitXYZ(RGBAImage const &imgX, RGBAImage const &imgY, RGBAImage const &imgZ)
  {
    ncvisual_options vopts{};
    vopts.flags = NCVISUAL_OPTION_NOINTERPOLATE;
    vopts.scaling = NCSCALE_SCALE;
    vopts.blitter = NCBLIT_PIXEL;

    vopts.n = pX;
    auto ncvX = ncvisual_from_rgba(imgX.data(), imgX.dimension(1), imgX.dimension(0) * 4, imgX.dimension(0));
    ncvisual_blit(nc, ncvX, &vopts);
    ncvisual_destroy(ncvX);

    vopts.n = pY;
    auto ncvY = ncvisual_from_rgba(imgY.data(), imgY.dimension(1), imgY.dimension(0) * 4, imgY.dimension(0));
    ncvisual_blit(nc, ncvY, &vopts);
    ncvisual_destroy(ncvY);

    vopts.n = pZ;
    auto ncvZ = ncvisual_from_rgba(imgZ.data(), imgZ.dimension(1), imgZ.dimension(0) * 4, imgZ.dimension(0));
    ncvisual_blit(nc, ncvZ, &vopts);
    ncvisual_destroy(ncvZ);
  }

  void setStatus(std::string const &status)
  {
    ncplane_cursor_move_yx(pStd, 0, 0);
    ncplane_printf(pStd, "%s", status.c_str());
  }

  void render() { notcurses_render(nc); }

  auto getInput() -> char32_t
  {
    char32_t input = 0;
    ncinput  ni;
    while (true) {
      input = notcurses_get_blocking(nc, &ni);
      if (input == (char32_t)-1) { continue; }
      if (ni.evtype == NCTYPE_RELEASE) { continue; }
      if (ni.evtype == NCTYPE_UNKNOWN) { continue; }
      break;
    }
    return input;
  }
};

int main_view(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to dump info from");
  args::ValueFlag<std::string>  basisFile(parser, "BASIS", "Basis file", {"basis", 'b'});
  // 's'); args::ValueFlag<char> component(parser, "X", "Component (m)agnitude, (p)hase, (r)eal, (i)maginary", {'c',
  // "component"}, 'm');

  ParseCommand(parser);
  HD5::Reader reader(iname.Get());
  Cx5         img = reader.readTensor<Cx5>("image");
  auto const  basis = ReadBasis(basisFile.Get());
  bool const  hasBasis = basis.dimension(1) > 1;
  if (hasBasis && basis.dimension(0) != img.dimension(0)) {
    Log::Fail("Basis length {} does not match number of images {}", basis.dimension(0), img.dimension(0));
  }

  Index const nB = (basis.dimension(1) > 1) ? basis.dimension(1) : img.dimension(0);
  Index const nI = img.dimension(1);
  Index const nJ = img.dimension(2);
  Index const nK = img.dimension(3);
  Index const nV = img.dimension(4);
  Index       iB = nB / 2;
  Index       iI = nI / 2;
  Index       iJ = nJ / 2;
  Index       iK = nK / 2;
  Index       iV = nV / 2;

  UI ui;

  bool  go = true;
  Index inc = 1;
  while (go) {
    Re2 sliceX = hasBasis ? Re2(basis.chip<1>(iB)
                                  .cast<Cx>()
                                  .contract(img.chip<4>(iV).chip<1>(iI), Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>())
                                  .abs())
                          : Re2(img.chip<4>(iV).chip<1>(iI).chip<0>(iB).abs());
    Re2 sliceY = hasBasis ? Re2(basis.chip<1>(iB)
                                  .cast<Cx>()
                                  .contract(img.chip<4>(iV).chip<2>(iJ), Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>())
                                  .abs())
                          : Re2(img.chip<4>(iV).chip<2>(iJ).chip<0>(iB).abs());
    Re2 sliceZ = hasBasis ? Re2(basis.chip<1>(iB)
                                  .cast<Cx>()
                                  .contract(img.chip<4>(iV).chip<3>(iK), Eigen::IndexPairList<Eigen::type2indexpair<0, 0>>())
                                  .abs())
                          : Re2(img.chip<4>(iV).chip<3>(iK).chip<0>(iB).abs());

    sliceX = sliceX / Maximum(sliceX);
    sliceY = sliceY / Maximum(sliceY);
    sliceZ = sliceZ / Maximum(sliceZ);

    auto const imgX = ToRGBA(sliceX);
    auto const imgY = ToRGBA(sliceY);
    auto const imgZ = ToRGBA(sliceZ);

    ui.blitXYZ(imgX, imgY, imgZ);
    ui.setStatus(
      fmt::format("{} B {}/{} I {}/{} J {}/{} K {}/{} V {}/{}", iname.Get(), iB, nB, iI, nI, iJ, nJ, iK, nK, iV, nV));
    ui.render();
    auto const input = ui.getInput();

    if (input == 'x') { break; }
    switch (input) {
    case 'q': iB = std::max(iB - inc, 0L); break;
    case 'a': iB = std::min(iB + inc, nB - 1); break;
    case 'w': iI = std::max(iI - inc, 0L); break;
    case 's': iI = std::min(iI + inc, nI - 1); break;
    case 'e': iJ = std::max(iJ - inc, 0L); break;
    case 'd': iJ = std::min(iJ + inc, nJ - 1); break;
    case 'r': iK = std::max(iK - inc, 0L); break;
    case 'f': iK = std::min(iK + inc, nK - 1); break;
    case 't': iV = std::max(iV - inc, 0L); break;
    case 'g': iV = std::min(iV + inc, nV - 1); break;
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

  return ui.stop();
}
