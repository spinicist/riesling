function [kspace, traj, info] = read_riesling(fname, vol)
% READ_RIESLING reads radial k-space data from riesling .h5 file
%
% Inputs:
%   - fname: file name
%   - vol: Volume to read (optional, default=-1 which reads all)
%
% Outputs:
%   - data: Radial k-space data
%   - traj: Trajectory
%   - info: Info structure
%
% Emil Ljungberg, King's College London, 2021

if nargin < 2
    vol = -1;
end

info = h5read(fname, '/info');
traj = h5read(fname, '/traj');
data_info = h5info(fname, '/volumes');

% Assuming same size for all volumes
nvol = length(data_info.Datasets);
if vol == -1
    volumes = 1:nvol;
else
    volumes = vol;
end

dsize = data_info.Datasets(1).Dataspace.Size;
kspace = zeros(dsize(1), dsize(2), dsize(3), nvol);

for i=volumes
    data = h5read(fname, sprintf('/volumes/%04d',i-1));
    kspace(:,:,:,i) = data.r + 1j*data.i;
end

end