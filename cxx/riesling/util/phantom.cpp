
#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/recon.hpp"
#include "rl/phantom/cartesian.hpp"
#include "rl/phantom/gradcubes.hpp"
#include "rl/phantom/radial.hpp"
#include "rl/phantom/shepp-logan.hpp"
#include "rl/phantom/sphere.hpp"
#include "rl/sense/sense.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

Trajectory LoadTrajectory(std::string const &file)
{
  Log::Print("Phan", "Reading external trajectory from {}", file);
  HD5::Reader reader(file);
  return Trajectory(reader, reader.readStruct<Info>(HD5::Keys::Info).voxel_size);
}

Trajectory CreateTrajectory(Index const matrix,
                            float const voxSz,
                            float const readOS,
                            Index const spokes,
                            Index const sps,
                            bool const  cart,
                            bool const  phyllo)
{
  Log::Print("Phan", "Using {} hi-res spokes", spokes);
  auto points = cart     ? Cartesian(matrix)
                : phyllo ? Phyllotaxis(matrix, readOS, spokes, 7, sps)
                         : ArchimedeanSpiral(matrix, readOS, spokes);
  Log::Print("Phan", "Samples: {} Traces: {}", points.dimension(1), points.dimension(2));
  return Trajectory(points, Sz3{matrix, matrix, matrix}, Eigen::Array3f::Constant(voxSz));
}

void main_phantom(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Filename to write phantom data to");

  args::ValueFlag<std::string> trajfile(parser, "TRAJ FILE", "Input HD5 file for trajectory", {"traj"});

  args::ValueFlag<float> voxSize(parser, "V", "Voxel size in mm (default 2)", {'v', "vox-size"}, 2.f);
  args::ValueFlag<Index> matrix(parser, "M", "Matrix size (default 128)", {'m', "matrix"}, 128);
  args::ValueFlag<float> size(parser, "SZ", "Phantom size/radius in mm (default 90)", {"size"}, 90.f);
  args::ValueFlag<Eigen::Vector3f, Vector3fReader> center(parser, "C", "Center of phantom in mm", {"center"});

  args::Flag gradCubes(parser, "C", "Grad cubes phantom", {"gradcubes"});
  args::Flag sphere(parser, "S", "Spherical phantom", {"sphere"});
  args::Flag hemi(parser, "H", "Hemispherical phantom", {"hemi"});

  args::Flag             cart(parser, "", "Use a Cartesian trajectory", {'c', "cart"});
  args::Flag             phyllo(parser, "", "Use a phyllotaxis", {'p', "phyllotaxis"});
  args::ValueFlag<Index> smoothness(parser, "S", "Phyllotaxis smoothness", {"smoothness"}, 10);
  args::ValueFlag<Index> spi(parser, "N", "Phyllotaxis segments per interleave", {"spi"}, 4);
  args::Flag             gmeans(parser, "N", "Golden-Means phyllotaxis", {"gmeans"});

  args::ValueFlag<float> readOS(parser, "S", "Read-out oversampling (2)", {'r', "read"}, 2);
  args::ValueFlag<Index> spokes(parser, "S", "Spokes", {"spokes"}, 2048);
  args::ValueFlag<Index> sps(parser, "S", "Spokes per segment", {"sps"}, 256);
  args::ValueFlag<float> snr(parser, "SNR", "Add noise (specified as SNR)", {'n', "snr"}, 0);

  ParseCommand(parser, iname);
  auto const       cmd = parser.GetCommand().Name();
  Trajectory const traj =
    trajfile ? LoadTrajectory(trajfile.Get())
             : CreateTrajectory(matrix.Get(), voxSize.Get(), readOS.Get(), spokes.Get(), sps.Get(), cart, phyllo);
  Info const  info{.voxel_size = Eigen::Array3f::Constant(voxSize.Get()),
                   .origin = Eigen::Array3f::Constant(-(voxSize.Get() * matrix.Get()) / 2.f),
                   .direction = Eigen::Matrix3f::Identity(),
                   .tr = 1.f};
  HD5::Writer writer(iname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  traj.write(writer);

  Cx3 phantom(traj.matrix());

  if (sphere) {
    phantom = SphericalPhantom(traj.matrix(), traj.voxelSize(), center.Get(), size.Get(), 1.f, false);
  } else if (hemi) {
    phantom = SphericalPhantom(traj.matrix(), traj.voxelSize(), center.Get(), size.Get(), 1.f, true);
  } else if (gradCubes) {
    phantom = GradCubes(traj.matrix(), traj.voxelSize(), size.Get());
  } else {
    // Pars for the 10 elipsoids in the 3D Shepp-Logan phantom from Cheng et al.
    std::vector<Eigen::Vector3f> const centres{{0, 0, 0},
                                               {0, 0, 0},
                                               {-0.22, 0, -0.25},
                                               {0.22, 0, -0.25},
                                               {0, 0.35, -0.25},
                                               {0, 0.1, -0.25},
                                               {-0.08, -0.65, -0.25},
                                               {0.06, -0.65, -0.25},
                                               {0.06, -0.105, 0.625},
                                               {0, 0.1, 0.625}};

    // Half-axes
    std::vector<Eigen::Array3f> const ha{{0.69, 0.92, 0.9},  {0.6624, 0.874, 0.88}, {0.41, 0.16, 0.21},   {0.31, 0.11, 0.22},
                                         {0.21, 0.25, 0.5},  {0.046, 0.046, 0.046}, {0.046, 0.023, 0.02}, {0.046, 0.023, 0.02},
                                         {0.056, 0.04, 0.1}, {0.056, 0.056, 0.1}};
    std::vector<float> const          angles{0, 0, 3 * M_PI / 5, 2 * M_PI / 5, 0, 0, 0, M_PI / 2, M_PI / 2, 0};
    std::vector<float> const          ints{100, -40, -10, -10, 10, 10, 5, 5, 10, -10};
    phantom = SheppLoganPhantom(traj.matrix(), traj.voxelSize(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), size.Get(),
                                centres, ha, angles, ints);
  }
  writer.writeTensor(HD5::Keys::Data, AddBack(phantom.dimensions(), 1, 1), phantom.data(), HD5::Dims::Images);
  Log::Print(cmd, "Finished");
}
