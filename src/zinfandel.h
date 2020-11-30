#pragma once

#include "log.h"
#include "radial.h"
#include "sense.h"

enum struct ZMode
{
  Z1_iter,
  Z1_simul
};

/*!
 * ZTE Infilling From Autocallibration NeighbourhooD ELements
 */
void zinfandel(
    ZMode const mode,
    long const gap_sz,
    long const n_src,
    long const n_cal_in,
    float const lambda,
    Cx3 &ks,
    Log &log);

/*!
 *  Helper functions exposed for testing
 */
Eigen::MatrixXcd GrabSources(
    Cx3 const &ks,
    float const scale,
    long const spoke,
    long const st_spoke,
    long const n_src,
    long const n_cal);

Eigen::MatrixXcd GrabSourcesOne(
    Cx3 const &ks,
    float const scale,
    long const spoke,
    long const st_spoke,
    long const n_src,
    long const n_cal);

Eigen::MatrixXcd GrabTargets(
    Cx3 const &ks,
    float const scale,
    long const spoke,
    long const st_spoke,
    long const n_tgt,
    long const n_cal);

void FillIterative(
    Eigen::VectorXcd const &S,
    Eigen::MatrixXcd const &W,
    float const scale,
    long const spoke,
    long const n_tgt,
    Cx3 &ks);

void FillSimultaneous(
    Eigen::VectorXcd const &S,
    Eigen::MatrixXcd const &W,
    float const scale,
    long const spoke,
    Cx3 &ks);

Eigen::MatrixXcd
CalcWeights1(Eigen::MatrixXcd const &src, Eigen::MatrixXcd const tgt, float const lambda);
Eigen::MatrixXcd
CalcWeights2(Eigen::MatrixXcd const &src, Eigen::MatrixXcd const tgt, float const lambda);