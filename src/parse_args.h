#pragma once

#include "info.h"
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

extern void WriteOutput(
    Cx4 const &vols,
    bool const mag,
    bool const needsSwap,
    Info const &info,
    std::string const &iname,
    std::string const &oname,
    std::string const &suffix,
    std::string const &ext,
    Log &log);

void WriteBasisVolumes(
    Cx5 const &basisVols,
    R2 const &basis,
    bool const mag,
    Info const &info,
    std::string const &iname,
    std::string const &oname,
    std::string const &suffix,
    std::string const &ext,
    Log &log);

long LastOrVal(args::ValueFlag<long> &sFlag, long const vols);

#define CORE_RECON_ARGS                                                                            \
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");                           \
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});      \
  args::ValueFlag<float> osamp(parser, "OSAMP", "Grid oversampling factor (2)", {'s', "os"}, 2.f); \
  args::Flag kb(parser, "KB", "Use Kaiser-Bessel interpolation", {"kb"});                          \
  args::Flag fastgrid(                                                                             \
      parser, "FAST", "Enable fast but thread-unsafe gridding", {"fast-grid", 'f'});               \
  args::ValueFlag<std::string> sdc(                                                                \
      parser, "SDC", "SDC type: 'pipe', 'radial', 'none', or filename", {"sdc"}, "pipe");          \
  args::ValueFlag<float> sdc_exp(                                                                  \
      parser, "SDC Exponent", "SDC Exponent (default 1.0)", {'e', "sdc_exp"}, 1.0f);               \
  args::ValueFlag<std::string> oftype(                                                             \
      parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");

#define COMMON_RECON_ARGS                                                                          \
  CORE_RECON_ARGS                                                                                  \
  args::ValueFlag<float> out_fov(                                                                  \
      parser, "OUT FOV", "Final FoV in mm (default header value)", {"fov"}, -1);                   \
  args::ValueFlag<float> tukey_s(                                                                  \
      parser, "TUKEY START", "Start-width of Tukey filter", {"tukey_start"}, 1.0f);                \
  args::ValueFlag<float> tukey_e(                                                                  \
      parser, "TUKEY END", "End-width of Tukey filter", {"tukey_end"}, 1.0f);                      \
  args::ValueFlag<float> tukey_h(                                                                  \
      parser, "TUKEY HEIGHT", "End height of Tukey filter", {"tukey_height"}, 0.0f);               \
  args::Flag mag(parser, "MAGNITUDE", "Output magnitude images only", {"mag", 'm'});
