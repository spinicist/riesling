#include "vc.h"

#include "cropper.h"
#include "fft3n.h"
#include "tensorOps.h"

#include <Eigen/SVD>

Cx4 VBC(Cx4 &maps, Log &log)
{
  long const nc = maps.dimension(0);
  long const nx = maps.dimension(1);
  long const ny = maps.dimension(2);
  long const nz = maps.dimension(3);

  Eigen::Map<Eigen::MatrixXcf const> mat(maps.data(), nc, nx * ny * nz);
  log.info("VBC SVD size {}x{}", mat.rows(), mat.cols());
  auto const svd = mat.bdcSvd(Eigen::ComputeThinV);
  Cx4 body(maps.dimensions());
  Eigen::Map<Eigen::MatrixXcf> bodymat(body.data(), nc, nx * ny * nz);
  bodymat = svd.matrixV().transpose();
  log.image(maps, "vbc-maps.nii");
  log.image(body, "vbc-body.nii");
  bodymat = bodymat.array().conjugate() / bodymat.array().abs();
  return body;
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

Cx3 Hammond(Cx4 const &maps, Log &log)
{
  long const nc = maps.dimension(0);
  long const nx = maps.dimension(1);
  long const ny = maps.dimension(2);
  long const nz = maps.dimension(3);
  log.info("Combining images via the Hammond method");

  long const refSz = 9;
  Cropper refCrop(Dims3{nx, ny, nz}, Dims3{refSz, refSz, refSz}, log);
  Cx1 const ref = refCrop.crop4(maps).sum(Sz3{1, 2, 3}).conjugate() /
                  refCrop.crop4(maps).sum(Sz3{1, 2, 3}).abs();

  using FixedOne = Eigen::type2index<1>;
  Eigen::IndexList<int, FixedOne, FixedOne, FixedOne> rsh;
  rsh.set(0, nc);
  Eigen::IndexList<FixedOne, int, int, int> brd;
  brd.set(1, nx);
  brd.set(2, ny);
  brd.set(3, nz);
  auto const broadcasted = ref.reshape(rsh).broadcast(brd);
  Cx3 const combined = (maps * broadcasted).sum(Sz1{0});
  Cx3 const rss = maps.square().sum(Sz1{0}).sqrt();
  log.image(combined, "hammond-combined.nii");
  log.image(rss, "hammond-rss.nii");

  return combined;
}