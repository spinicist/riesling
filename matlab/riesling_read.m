function varargout = riesling_read(fname, dset)
% READ_RIESLING reads radial k-space or image data from riesling .h5 file
%
% Inputs:
%   - fname: file name
%    - dset: dataset name (default = 'data')
%
% Outputs:
%   - data: Complex valued data
%   - info: Info structure (if present)
%   - traj: Trajectory (if present)
%
% Emil Ljungberg, King's College London, 2021
% Martin Krämer, University Hospital Jena, 2021
% Patrick Fuchs, University College London, 2022
% David Leitao, King's College London, 2024

if nargin < 2
    dset = 'data';
end

file_info = h5info(fname);
data = h5read(fname, strcat('/', dset));

if strcmpi(dset,'data')
    data = data.r + 1j*data.i;
end
varargout{1} = data;

has_info = any(contains({file_info.Datasets.Name}, 'info'));
if has_info
    info = h5read(fname, '/info');
    varargout{2} = info;
end

has_trajectory = any(contains({file_info.Datasets.Name}, 'trajectory'));
if has_trajectory
    % load k-space data and trajectory
    traj = h5read(fname, '/trajectory');
    varargout{3} = traj;
end
end
