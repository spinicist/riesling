#pragma once

#include <args.hxx>

extern args::Group    global_group;
extern args::HelpFlag help;
extern args::Flag     verbose;

void SetLogging(std::string const &name);
void ParseCommand(args::Subparser &parser);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname, args::Positional<std::string> &oname);
