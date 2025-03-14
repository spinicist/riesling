#include "args.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"
#include "merlin.hpp"

extern args::Group    global_group;
extern args::HelpFlag help;
extern args::Flag     verbose;

using namespace rl;

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser          parser("MERLIN");
  args::GlobalOptions           globals(parser, global_group);
  args::Positional<std::string> iname(parser, "INPUT", "ifile HD5 file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output HD5 file");
  args::ValueFlag<std::string>  mname(parser, "MASK", "Mask HD5 file", {'m', "mask"});

  try {
    parser.ParseCLI(argc, argv);
    SetLogging("MERLIN");
    if (!iname) { throw args::Error("No input file specified"); }
    if (!oname) { throw args::Error("No output file specified"); }

    HD5::Reader ifile(iname.Get());
    HD5::Writer ofile(oname.Get());
    merlin::ImageType::Pointer mask = nullptr;
    Re3 mdata;
    if (mname) {
      HD5::Reader mfile(mname.Get());
      mdata = mfile.readTensor<Re3>();
      mask = merlin::Import(mdata, mfile.readInfo());
    }
    Re4        idata = ifile.readTensor<Cx5>().abs().chip<4>(0); // ITK does not like this being const
    auto const info = ifile.readInfo();
    auto       tfm = merlin::TransformType::New();
    auto const fixed = merlin::Import(ChipMap(idata, 0), info);
    for (Index ii = 1; ii < idata.dimension(3); ii++) {
      auto const moving = merlin::Import(ChipMap(idata, ii), info);
      Log::Print("MERLIN", "Register navigator {} to {}", ii, 0);
      auto tfm = merlin::Register(fixed, moving, mask);
      ofile.writeTransform(merlin::ITKToRIESLING(tfm), fmt::format("{:02d}", ii));
    }
    Log::Print("MERLIN", "Finished");
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, "{}\n", parser.Help());
    return EXIT_SUCCESS;
  } catch (args::Error &e) {
    fmt::print(stderr, "{}\n", parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", e.what());
    return EXIT_FAILURE;
  } catch (Log::Failure &f) {
    Log::Fail(f);
    Log::End();
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    Log::Fail(Log::Failure("None", "{}", e.what()));
    Log::End();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
