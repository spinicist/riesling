#pragma once

#include <args.hxx>
#include <vector>

#include "types.h"

extern args::Group global_group;
extern args::HelpFlag help;
extern args::Flag verbose;

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname);
void ParseCommand(args::Subparser &parser);

struct Vector3fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Vector3f &x);
};

template <typename T>
struct VectorReader
{
  void operator()(std::string const &name, std::string const &value, std::vector<T> &x);
};

struct Sz3Reader
{
  void operator()(std::string const &name, std::string const &value, Sz3 &x);
};

// Helper function to generate a good output name
std::string OutName(
  std::string const &iName,
  std::string const &oName,
  std::string const &suffix,
  std::string const &extension = "h5");

// Helper function for getting a good volume to take SENSE maps from
Index ValOrLast(Index const val, Index const last);

#define CORE_RECON_ARGS                                                                            \
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");                           \
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});      \
  args::ValueFlag<float> osamp(parser, "OSAMP", "Grid oversampling factor (2)", {'s', "os"}, 2.f); \
  args::ValueFlag<std::string> ktype(                                                              \
    parser, "K", "Choose kernel - NN, KB3, KB5", {'k', "kernel"}, "KB3");                          \
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
