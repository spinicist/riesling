#pragma once

#include "info.h"
#include "kernel.h"
#include "log.h"
#include "sdc.h"
#include <args.hxx>
#include <vector>

extern args::Group global_group;
extern args::HelpFlag help;
extern args::Flag verbose;

Log ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname);
Log ParseCommand(args::Subparser &parser);

struct Vector3fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Vector3f &x);
};

struct VectorReader
{
  void operator()(std::string const &name, std::string const &value, std::vector<float> &x);
};

std::string OutName(
  std::string const &iName,
  std::string const &oName,
  std::string const &suffix,
  std::string const &extension = "nii");

#define CORE_RECON_ARGS                                                                            \
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");                           \
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});      \
  args::ValueFlag<float> osamp(parser, "OSAMP", "Grid oversampling factor (2)", {'s', "os"}, 2.f); \
  std::unordered_map<std::string, Kernels> kernelMap{                                              \
    {"NN", Kernels::NN}, {"KB3", Kernels::KB3}, {"KB5", Kernels::KB5}};                            \
  args::MapFlag<std::string, Kernels> kernel(                                                      \
    parser, "K", "Choose kernel - NN, KB3, KB5", {'k', "kernel"}, kernelMap);                      \
  args::Flag fastgrid(                                                                             \
    parser, "FAST", "Enable fast but thread-unsafe gridding", {"fast-grid", 'f'});                 \
  args::ValueFlag<std::string> sdc(                                                                \
    parser, "SDC", "SDC type: 'pipe', 'pipenn', 'none', or filename", {"sdc"}, "pipenn");          \
  args::ValueFlag<float> sdcPow(parser, "P", "SDC Power (default 1.0)", {"sdcPow"}, 1.0f);

#define COMMON_RECON_ARGS                                                                          \
  CORE_RECON_ARGS                                                                                  \
  args::ValueFlag<float> out_fov(                                                                  \
    parser, "OUT FOV", "Final FoV in mm (default header value)", {"fov"}, -1);                     \
  args::ValueFlag<float> tukey_s(                                                                  \
    parser, "TUKEY START", "Start-width of Tukey filter", {"tukey_start"}, 1.0f);                  \
  args::ValueFlag<float> tukey_e(                                                                  \
    parser, "TUKEY END", "End-width of Tukey filter", {"tukey_end"}, 1.0f);                        \
  args::ValueFlag<float> tukey_h(                                                                  \
    parser, "TUKEY HEIGHT", "End height of Tukey filter", {"tukey_height"}, 0.0f);                 \
  args::Flag mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
