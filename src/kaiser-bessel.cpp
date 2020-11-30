#include "fmt/format.h"
#include "kaiser-bessel.h"

#include <cmath>
#include <vector>

static std::vector<float> const coeff_0to8{
    0.143432f,
    0.144372f,
    0.147260f,
    0.152300f,
    0.159883f,
    0.170661f,
    0.185731f,
    0.207002f,
    0.238081f,
    0.286336f,
    0.366540f,
    0.501252f,
    0.699580f,
    0.906853f,
    1.000000f,
};

static std::vector<float> const coeff_8toinf{
    0.405687f, 0.405664f, 0.405601f, 0.405494f, 0.405349f, 0.405164f, 0.404945f, 0.404692f,
    0.404413f, 0.404107f, 0.403782f, 0.403439f, 0.403086f, 0.402724f, 0.402359f, 0.401995f,
    0.401637f, 0.401287f, 0.400951f, 0.400631f, 0.400332f, 0.400055f, 0.399805f, 0.399582f,
    0.399391f, 0.399231f, 0.399106f, 0.399012f, 0.398998f, 0.399001f};

float chebeval(float x, std::vector<float> const &pval)
{
  float norm = 0.;
  float val = 0.;

  for (size_t i = 0; i < pval.size(); i++) {

    float dist = x - cosf(M_PI * (float)i / (float)(pval.size() - 1));

    if (0. == dist)
      return pval[i];

    float weight = ((0 == i % 2) ? 1. : -1.) / dist;

    if ((0 == i) || (pval.size() - 1 == i))
      weight /= 2.;

    norm += weight;
    val += weight * pval[i];
  }

  return val / norm;
}

/*
 * modified bessel function
 */
float bessel_i0(float const &x)
{
  auto const abs_x = fabs(x);
  if (x < 8.)
    return exp(abs_x) * chebeval(x / 4. - 1., coeff_0to8);

  return exp(abs_x) * chebeval(16. / x - 1., coeff_8toinf) / sqrt(abs_x);
}

float KB(float const &beta, float const &x, float const &w)
{
  return bessel_i0(beta * sqrt(1. - pow(2. * x / w, 2.))) / bessel_i0(beta);
}

float KB_FT(float const &beta, float const &x, float const &w)
{
  double const a = sqrt(pow(beta, 2.) - pow(M_PI * x * w, 2.));
  return a / sinh(a); // * bessel_i0(beta);
}
