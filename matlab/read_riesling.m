function [kspace, traj, info] = read_riesling(fname)
% READ_RIESLING reads radial k-space data from riesling .h5 file
%
% Inputs:
%   - fname: file name
%
% Outputs:
%   - data: Radial k-space data
%   - traj: Trajectory
%   - info: Info structure
%
% Emil Ljungberg, King's College London, 2021

info = h5read(fname, '/info');
traj = h5read(fname, '/trajectory');
data = h5read(fname, '/noncartesian');
kspace = data.r + 1j*data.i;

end