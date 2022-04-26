#include "types.h"

#include "io/io.h"
#include "log.h"
#include "parse_args.h"
#include "tensorOps.h"

int main_downsamp(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<float> dsamp(parser, "D", "Downsample by factor", {"ds"}, 2.0);
  args::ValueFlag<Index> lores(parser, "L", "First N spokes are lo-res", {"lores"}, 0);
  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  auto traj = reader.trajectory();
  Info const info = traj.info();
  Info dsInfo = info;

  if (lores) {
    Log::Print(FMT_STRING("Ignoring {} lo-res spokes"), lores.Get());
  }

  // Account for rounding
  dsInfo.matrix = (info.matrix.cast<float>() / dsamp.Get()).cast<Index>();
  float const ds = static_cast<float>(info.matrix[0]) / dsInfo.matrix[0];
  dsInfo.voxel_size = info.voxel_size * ds;
  Index sz = 3; // Need this for slicing below
  if (dsInfo.type == Info::Type::ThreeDStack) {
    dsInfo.matrix[2] = info.matrix[2];
    dsInfo.voxel_size[2] = info.voxel_size[2];
    sz = 2;
  }
  Index minRead = info.read_points, maxRead = 0;
  R3 dsPoints(traj.points().dimensions());
  for (Index is = 0; is < info.spokes; is++) {
    for (Index ir = 0; ir < info.read_points; ir++) {
      R1 p = traj.points().chip<2>(is).chip<1>(ir);
      p.slice(Sz1{0}, Sz1{sz}) *= p.slice(Sz1{0}, Sz1{sz}).constant(ds);
      if (Norm(p.slice(Sz1{0}, Sz1{sz})) <= 0.5f) {
        dsPoints.chip<2>(is).chip<1>(ir) = p;
        if (is >= lores.Get()) { // Ignore lo-res spokes for this calculation
          minRead = std::min(minRead, ir);
          maxRead = std::max(maxRead, ir);
        }
      } else {
        dsPoints.chip<2>(is).chip<1>(ir).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  dsInfo.read_points = 1 + maxRead - minRead;
  Log::Print(
    FMT_STRING("Downsampling by {}, new voxel-size {} matrix {}, kept read-points {}-{}"),
    ds,
    dsInfo.voxel_size.transpose(),
    dsInfo.matrix.transpose(),
    minRead,
    maxRead);
  dsPoints = R3(dsPoints.slice(Sz3{0, minRead, 0}, Sz3{3, dsInfo.read_points, dsInfo.spokes}));
  Cx4 ks = reader.readTensor<Cx4>(HD5::Keys::Noncartesian)
             .slice(
               Sz4{0, minRead, 0, 0},
               Sz4{dsInfo.channels, dsInfo.read_points, dsInfo.spokes, dsInfo.volumes});

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "downsamp"));
  writer.writeTrajectory(Trajectory(dsInfo, dsPoints, traj.frames()));
  writer.writeTensor(ks, HD5::Keys::Noncartesian);

  return EXIT_SUCCESS;
}