#include "vc.h"

#include "fft3n.h"
#include "tensorOps.h"

#include <Eigen/SVD>

void VBC(Cx4 &maps, Log &log)
{
  long const nc = maps.dimension(0);
  long const nx = maps.dimension(1);
  long const ny = maps.dimension(2);
  long const nz = maps.dimension(3);

  Eigen::Map<Eigen::MatrixXcf const> mat(maps.data(), nc, nx * ny * nz);
  log.info("VBC SVD size {}x{}", mat.rows(), mat.cols());
  auto const svd = mat.bdcSvd(Eigen::ComputeThinV);
  Cx3 body(nx, ny, nz);
  Eigen::Map<Eigen::VectorXcf> bodymat(body.data(), nx * ny * nz);
  bodymat = svd.matrixV().col(0);
  bodymat = bodymat.array() / bodymat.array().abs();

  log.image(body, "vbc-body.nii");
  maps = maps / Tile(body, nc);
  log.image(maps, "vbc-corrected.nii");
}

void VCC(Cx4 &data, Log &log)
{
  long const nc = data.dimension(0);
  long const nx = data.dimension(1);
  long const ny = data.dimension(2);
  long const nz = data.dimension(3);

  // Assemble our virtual conjugate channels
  Cx4 cdata(nc, nx, ny, nz);
  FFT3N fft(cdata, log);
  cdata = data;
  log.image(cdata, "vcc-cdata.nii");
  fft.forward(cdata);
  log.image(cdata, "vcc-cdata-ks.nii");
  Cx4 rdata = cdata.slice(Sz4{0, 1, 1, 1}, Sz4{nc, nx - 1, ny - 1, nz - 1})
                  .reverse(Eigen::array<bool, 4>({false, true, true, true}))
                  .conjugate();
  cdata.setZero();
  cdata.slice(Sz4{0, 1, 1, 1}, Sz4{nc, nx - 1, ny - 1, nz - 1}) = rdata;
  log.image(cdata, "vcc-cdata-conj-ks.nii");
  fft.reverse(cdata);
  log.image(cdata, "vcc-cdata-conj.nii");

  Cx3 phase(nx, ny, nz);
  phase.setZero();
  for (long iz = 1; iz < nz; iz++) {
    for (long iy = 1; iy < ny; iy++) {
      for (long ix = 1; ix < nx; ix++) {
        Cx1 const vals = data.chip(iz, 3).chip(iy, 2).chip(ix, 1);
        Cx1 const cvals = cdata.chip(iz, 3).chip(iy, 2).chip(ix, 1).conjugate(); // Dot has a conj
        float const p = std::log(Dot(cvals, vals)).imag() / 2.f;
        phase(ix, iy, iz) = std::polar(1.f, -p);
      }
    }
  }
  log.image(phase, "vcc-correction.nii");
  log.info("Applying Virtual Conjugate Coil phase correction");
  data = data * Tile(phase, nc);
  log.image(data, "vcc-corrected.nii");
}
