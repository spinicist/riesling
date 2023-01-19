function varargout = riesling_read(fname)
% READ_RIESLING reads radial k-space or image data from riesling .h5 file
%
% Inputs:
%   - fname: file name
%
% Outputs:
%   can be any of the following (depending on the input data)
%   - data: Radial k-space data or image data
%   - info: Info structure
%   - traj: Trajectory (only in case of k-space data)
%   - sense: Sense data
%
% Emil Ljungberg, King's College London, 2021
% Martin Krämer, University Hospital Jena, 2021
% Patrick Fuchs, University College London, 2022

% Open h5 file
file_info = h5info(fname);
is_img_data = any(contains({file_info.Datasets.Name}, 'image'));
is_sense_data = any(contains({file_info.Datasets.Name}, 'sense'));
is_nufft_reverse_data = any(contains({file_info.Datasets.Name}, 'nufft-backward'));
is_nufft_forward_data = any(contains({file_info.Datasets.Name}, 'nufft-forward'));

% read info 
if ~is_sense_data
    info = h5read(fname, '/info');
end

if is_img_data % load only image data
    data = h5read(fname, '/image');
    img = data.r + 1j*data.i;
    
    varargout{1} = img;
    varargout{2} = info;
elseif is_nufft_forward_data
    data = h5read(fname, '/nufft-forward');
    img = data.r + 1j*data.i;
    
    varargout{1} = img;
    varargout{2} = info;
elseif is_nufft_reverse_data
    data = h5read(fname, '/nufft-backward');
    img = data.r + 1j*data.i;
    
    varargout{1} = img;
    varargout{2} = info;
elseif is_sense_data
    data = h5read(fname, '/sense');
    img = data.r + 1j*data.i;
    
    varargout{1} = img;
else % load k-space data and trajectory
    traj = h5read(fname, '/trajectory');
    data = h5read(fname, '/noncartesian');
    kspace = data.r + 1j*data.i;

    varargout{1} = kspace;
    varargout{2} = traj;
    varargout{3} = info;
end

end
