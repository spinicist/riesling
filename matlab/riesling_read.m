function varargout = riesling_read(fname)
% READ_RIESLING reads radial k-space or image data from riesling .h5 file
%
% Inputs:
%   - fname: file name
%
% Outputs:
%   - data: Radial k-space data or image data
%   - info: Info structure
%   - traj: Trajectory (only in case of k-space data)
%
% Emil Ljungberg, King's College London, 2021
% Martin Krämer, University Hospital Jena, 2021

% get h5 file info and check if image data is present
file_info = h5info(fname);
is_img_data = any(contains({file_info.Datasets.Name}, 'image'));

if is_img_data % load only image data
    data = h5read(fname, '/image');
    img = data.r + 1j*data.i;
    
    varargout{1} = img;
else % load k-space data and trajectory
    traj = h5read(fname, '/trajectory');
    data = h5read(fname, '/noncartesian');
    kspace = data.r + 1j*data.i;

    varargout{1} = kspace;
    varargout{3} = traj;
end

% always read info 
info = h5read(fname, '/info');
varargout{2} = info;

end