#include "cg.h"
#include "fft.h"
#include "threads.h"

inline auto sign(Cx3 const &x) -> Cx3
{
  return (x != std::complex<float>(0.f, 0.f)).select(x / x.abs(), x.constant(0.f));
};
inline auto shrink(Cx3 const &x, float const y) -> Cx3
{
  return sign(x) * (x.abs() - y).cwiseMax(0.f);
};

Cx3 cgisR0(
    Cx3 &radial,
    Cx3::Dimensions const &dims,
    EncodeFunction const &encode_ks,
    DecodeFunction const &decode_ks,
    long const &max_its,
    float const &thresh,
    float const &mu, // l1-weight
    Log &log)
{
  // Allocate all memory
  Cx3 g(dims);
  Cx3 r(dims);
  Cx3 resid(radial.dimensions());
  Cx3 u(dims);
  decode_ks(radial, u);
  R0 const scale = u.abs().maximum();
  u = (1.f / scale()) * u;
  radial = (1.f / scale()) * radial;
  encode_ks(u, resid);
  resid.device(Threads::GlobalDevice()) -= radial;

  float n2r = norm2(resid);
  R0 l1 = u.abs().sum();
  float const psi_0 = n2r / 2.f + mu * l1();
  float psi_prev = psi_0;
  float alpha_prev = std::numeric_limits<float>::infinity();
  log.info(
      FMT_STRING("Starting CGIST. Threshold {} Scale {} Norm {} L1 {} Psi {}"),
      thresh,
      scale(),
      n2r,
      l1(),
      psi_0);

  for (long icg = 0; icg < max_its; icg++) {
    // Calculate q = DEC/ENC p
    decode_ks(resid, g);

    r = (u.abs() > 0.f).select(-(g + mu * sign(u)), shrink(-g, mu));
    encode_ks(r, resid);
    float const alpha = dot(r, r) / dot(resid, resid);
    if (alpha > alpha_prev) {
      log.info(FMT_STRING("Alpha (step size) increased, stopping, should implement line search"));
      break;
    }
    u = shrink(u - alpha * g, alpha * mu);
    encode_ks(u, resid);
    resid.device(Threads::GlobalDevice()) -= radial;
    n2r = norm2(resid);
    l1 = u.abs().sum();
    float const psi_i = n2r / 2.f + mu * l1();
    float const delta = (psi_prev - psi_i);
    log.info(
        FMT_STRING("Step {} Alpha {} Norm {} l1 {} Psi {} Change {}"),
        icg,
        alpha,
        n2r,
        l1(),
        psi_i,
        delta);
    if (fabs(delta) < thresh) {
      break;
    }
    psi_prev = psi_i;
    alpha_prev = alpha;
  }
  u = scale() * u;
  return u;
}

Cx3 cgist(
    Cx3 &radial,
    Cx3::Dimensions const &dims,
    EncodeFunction const &encode_ks,
    DecodeFunction const &decode_ks,
    long const &max_its,
    float const &thresh,
    float const &mu, // l1-weight
    Log &log)
{
  // Allocate all memory. p1 stands for +1, m1 for -1, i.e. values at next/previous iterate
  Dims3 const rdims = radial.dimensions();
  Cx3 g(dims), gp1(dims), gm1(dims), ghat(dims);
  Cx3 r(dims), rm1(dims), rhat(dims);
  Cx3 ls(rdims), lsp1(rdims), lsm1(rdims), lshat(rdims);
  Cx3 Ar(rdims);
  Cx3 u(dims), up1(dims), um1(dims), uh(dims);
  float alpha;

  auto residual = [&radial, &encode_ks](Cx3 const &x, Cx3 &y) {
    encode_ks(x, y);
    y.device(Threads::GlobalDevice()) -= radial;
  };

  // Get first iterate values and scale problem
  decode_ks(radial, u);
  R0 const scale = u.abs().maximum();
  u = (1.f / scale()) * u;
  radial = (1.f / scale()) * radial;

  residual(u, ls);
  decode_ks(ls, g);
  r = (u.abs() > 0.f).select(-(g + mu * sign(u)), shrink(-g, mu));
  encode_ks(r, Ar);
  alpha = dot(r, r) / dot(Ar, Ar);

  up1 = shrink(u - alpha * g, alpha * mu);
  residual(up1, lsp1);
  decode_ks(lsp1, gp1);

  for (long icg = 2; icg < max_its; icg++) {
    // Store iterates
    rm1 = r;
    um1 = u;
    u = up1;
    lsm1 = ls;
    ls = lsp1;
    gm1 = g;
    g = gp1;
    float ls_cost = norm2(ls);
    R0 l1_cost = u.abs().sum();
    float psi = ls_cost / 2.f + mu * l1_cost();

    // New values
    r = (u.abs() > 0.f).select(-(g + mu * sign(u)), shrink(-g, mu));
    float r_dot_r = dot(r, r);
    R0 active_set_size = sign(u).abs().sum();
    float subGradNorm = sqrt(r_dot_r) / active_set_size();
    if (subGradNorm < thresh) {
      log.info(
          FMT_STRING("Sub-gradient norm {} reached threshold {}, terminating"),
          subGradNorm,
          thresh);
      break;
    }
    encode_ks(r, Ar);
    alpha = r_dot_r / dot(Ar, Ar);

    up1 = shrink(u - alpha * g, alpha * mu);
    uh = up1;

    float test = norm(sign(up1) - sign(u)) + norm(sign(um1) - sign(u));
    log.info(
        FMT_STRING("Sub-gradient norm {} Active Set Size {} Alpha {} LS {} l1 {} Psi {}"),
        subGradNorm,
        active_set_size(),
        alpha,
        ls_cost,
        l1_cost(),
        psi);
    if (test < 0.5f) {
      log.info(FMT_STRING("Active set not changing, taking conjugate-gradient step"));
      // Conjugate Gradient Step
      lshat = ls + alpha * Ar;
      decode_ks(lshat, ghat);
      rhat = r - (u != u.constant(0.f)).select(ghat - g, ghat.constant(0.f));
      float beta = dot(rm1, rhat) / dot(rm1, rm1);
      R0 top = (um1 != um1.constant(0.f)).select(up1 / um1, up1).abs().minimum();
      log.info(FMT_STRING("Beta {} Top {}"), beta, top());
      beta = std::min(beta, top());
      auto const &tbeta = uh.constant(beta); // A Tensor filled with beta
      up1 = (uh - tbeta * um1) / (1.f - tbeta);
      lsp1 = (lshat - tbeta * lsm1) / (1.f - tbeta);
      gp1 = (ghat - tbeta * gm1) / (1.f - tbeta);
    } else {
      // Active set changing, don't accelerate
      residual(up1, lsp1);
      decode_ks(lsp1, gp1);
    }

    float lsp1_cost = norm2(lsp1);
    R0 l1p1_cost = up1.abs().sum();
    float psi_p1 = lsp1_cost / 2.f + mu * l1p1_cost();

    while (((psi_p1 / psi - 1.f) > 0.01) && (alpha > 1.e-8)) {
      alpha = alpha / 2.f;
      up1 = shrink(u - alpha * g, alpha * mu);
      residual(up1, lsp1);
      lsp1_cost = norm2(lsp1);
      l1p1_cost = up1.abs().sum();
      psi_p1 = lsp1_cost / 2.f + mu * l1p1_cost();
      log.info(FMT_STRING("Alpha {} LS {} l1 {} Psi {}"), alpha, lsp1_cost, l1p1_cost(), psi_p1);
    }
    decode_ks(lsp1, gp1);
  }
  u = scale() * u;
  return u;
}