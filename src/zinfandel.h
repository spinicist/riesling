#pragma once

#include "log.h"
#include "radial.h"
#include "sense.h"

/*!
 * ZTE Infilling From Autocallibration NeighbourhooD ELements
 */
void zinfandel(
    long const gap_sz,
    long const n_tgt,
    long const n_src,
    long const n_spoke,
    long const n_read,
    float const lambda,
    Cx3 &ks,
    Log &log);

/*!
 *  Helper functions exposed for testing
 */
Eigen::MatrixXcd GrabSources(
    Cx3 const &ks,
    float const scale,
    long const n_src,
    long const s_spoke,
    long const n_spoke,
    long const s_read,
    long const n_read);

Eigen::MatrixXcd GrabTargets(
    Cx3 const &ks,
    float const scale,
    long const n_tgt,
    long const s_spoke,
    long const n_spoke,
    long const s_read,
    long const n_read);

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
