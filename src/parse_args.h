#pragma once

#include "log.h"
#include "radial.h"
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
extern void WriteVolumes(
    RadialInfo const &info, T const &vols, long const which, std::string const &fname, Log &log);