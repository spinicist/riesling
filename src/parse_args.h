#pragma once

#include "info.h"
#include "log.h"
#include <args.hxx>
#include <vector>

extern args::Group global_group;
extern args::HelpFlag help;
extern args::Flag verbose;

Log ParseCommand(args::Subparser &parser, args::Positional<std::string> &fname);

std::string OutName(
    args::Positional<std::string> &inName,
    args::ValueFlag<std::string> &name,
    std::string const &suffix,
    std::string const &extension = "nii");

long SenseVolume(args::ValueFlag<long> &sFlag, long const vols);
std::vector<long> WhichVolumes(long const which, long const max_volume);
template <typename T>
extern void
WriteVolumes(Info const &info, T const &vols, long const which, std::string const &fname, Log &log);

#define COMMON_RECON_ARGS                                                                          \
  args::Positional<std::string> fname(parser, "FILE", "HD5 file to recon");                        \
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});      \
                                                                                                   \
  args::ValueFlag<long> volume(parser, "VOLUME", "Only recon this volume", {"vol"}, -1);           \
  args::ValueFlag<float> out_fov(                                                                  \
      parser, "OUT FOV", "Final FoV in mm (default header value)", {"fov"}, -1);                   \
  args::ValueFlag<float> osamp(parser, "OSAMP", "Grid oversampling factor (2)", {'s', "os"}, 2.f); \
  args::Flag stack(parser, "STACK", "Trajectory is stack-of-stars or similar", {"stack"});         \
  args::Flag kb(parser, "KB", "Use Kaiser-Bessel interpolation", {"kb"});                          \
  args::ValueFlag<long> kw(                                                                        \
      parser, "KERNEL WIDTH", "Width of gridding kernel. Default 1 for NN, 3 for KB", {"kw"}, 3);  \
  args::Flag sdc(parser, "SDC", "Use Sample Density Compensation (Zwart/Pipe/Menon)", {"sdc"});    \
  args::ValueFlag<float> dc_exp(                                                                   \
      parser, "DC Exponent", "Density-Compensation Exponent (default 1.0)", {'d', "dce"}, 1.0f);   \
                                                                                                   \
  args::ValueFlag<float> tukey_s(                                                                  \
      parser, "TUKEY START", "Start-width of Tukey filter", {"tukey_start"}, 1.0f);                \
  args::ValueFlag<float> tukey_e(                                                                  \
      parser, "TUKEY END", "End-width of Tukey filter", {"tukey_end"}, 1.0f);                      \
  args::ValueFlag<float> tukey_h(                                                                  \
      parser, "TUKEY HEIGHT", "End height of Tukey filter", {"tukey_height"}, 0.0f);
