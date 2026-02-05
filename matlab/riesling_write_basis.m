function [] = riesling_write_basis(fname, basis, basis_labels, dynamics, dynamics_labels, projection, projection_labels)
% WRITE_RIESLING writes radial k-space data to riesling .h5 format
%
% Input:
%   - fname: output filename
%   - data: Complex kspace data [nchannels, nsamples, ntraces, nslab, nvol]
%   - traj: Trajectory [3, nsamples, ntraces]
%   - matrix: Nominal reconstruction matrix (Optional)
%   - info: Info struct containing 'voxel_size', 'origin', 'direction', 'tr' (Optional)
%   - dim_labels: Optional dimension labels for dataset
%
% Inspired by: https://stackoverflow.com/questions/46203309/write-complex-numbers-in-an-hdf5-dataset-with-matlab
% Emil Ljungberg, King's College London, 2021
% Patrick Fuchs, University College London, 2022
% David Leitao, King's College London, 2025

if isfile(fname)
    warning("%s already exists. Deleting current file before writing new one.\n", fname);
    delete(fname);
end

fprintf('Opening %s\n',fname);
file = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');


% Separate complex data
wbasis = struct;
wbasis.r = real(basis);
wbasis.i = imag(basis);
floatType = H5T.copy('H5T_NATIVE_FLOAT');
sz = [H5T.get_size(floatType), H5T.get_size(floatType)];
% Compute the offsets to each field. The first offset is always zero.
offset = [0, sz(1)];

% Create the compound datatype for the file and for the memory (same).
filetype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert(filetype, 'r', offset(1), floatType);
H5T.insert(filetype, 'i', offset(2), floatType);

Ndims = 3;
if ndims(basis) == 3
    dims = size(basis);
else
    dims = ones(1, 3);
    dims(1:ndims(basis)) = size(basis);
end
dims = fliplr(dims);

space = H5S.create_simple(Ndims, dims, []);
ncart = H5D.create(file, '/basis', filetype, space, 'H5P_DEFAULT');
H5D.write(ncart, filetype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', wbasis);

if nargin >=3 && ~isempty(basis_labels)
    basis_labels = fliplr(basis_labels);
    for n=1:numel(basis_labels)
        H5DS.set_label(ncart,n-1,basis_labels{n});
    end
end

% Close it all up
H5D.close(ncart);
H5S.close(space);
H5T.close(filetype);

if ~isempty(dynamics)

    % Separate complex data
    wdyn = struct;
    wdyn.r = real(dynamics);
    wdyn.i = imag(dynamics);
    floatType = H5T.copy('H5T_NATIVE_FLOAT');
    sz = [H5T.get_size(floatType), H5T.get_size(floatType)];
    % Compute the offsets to each field. The first offset is always zero.
    offset = [0, sz(1)];
    
    % Create the compound datatype for the file and for the memory (same).
    filetype = H5T.create ('H5T_COMPOUND', sum(sz));
    H5T.insert(filetype, 'r', offset(1), floatType);
    H5T.insert(filetype, 'i', offset(2), floatType);
    
    % Ndims = 2;
    % if ndims(dynamics) == 2
    %     dims = size(dynamics);
    % else
    %     dims = ones(1, 2);
    %     dims(1:ndims(dynamics)) = size(dynamics);
    % end
    % dims = fliplr(dims);
    dims = fliplr(size(dynamics));
    Ndims = numel(dims);
    
    space = H5S.create_simple(Ndims, dims, []);
    ncart = H5D.create(file, '/dynamics', filetype, space, 'H5P_DEFAULT');
    H5D.write(ncart, filetype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', wdyn);
    
    if nargin >=4 && ~isempty(dynamics_labels)
        dynamics_labels = fliplr(dynamics_labels);
        for n=1:numel(dynamics_labels)
            H5DS.set_label(ncart,n-1,dynamics_labels{n});
        end
    end
    
    % Close it all up
    H5D.close(ncart);
    H5S.close(space);
    H5T.close(filetype);
end

if ~isempty(projection)

    % Separate complex data
    wpro = struct;
    wpro.r = real(projection);
    wpro.i = imag(projection);
    floatType = H5T.copy('H5T_NATIVE_FLOAT');
    sz = [H5T.get_size(floatType), H5T.get_size(floatType)];
    % Compute the offsets to each field. The first offset is always zero.
    offset = [0, sz(1)];
    
    % Create the compound datatype for the file and for the memory (same).
    filetype = H5T.create ('H5T_COMPOUND', sum(sz));
    H5T.insert(filetype, 'r', offset(1), floatType);
    H5T.insert(filetype, 'i', offset(2), floatType);
    
    % Ndims = 2;
    % if ndims(dynamics) == 2
    %     dims = size(dynamics);
    % else
    %     dims = ones(1, 2);
    %     dims(1:ndims(dynamics)) = size(dynamics);
    % end
    % dims = fliplr(dims);
    dims = fliplr(size(projection));
    Ndims = numel(dims);
    
    space = H5S.create_simple(Ndims, dims, []);
    ncart = H5D.create(file, '/projection', filetype, space, 'H5P_DEFAULT');
    H5D.write(ncart, filetype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', wpro);
    
    if nargin >=6 && ~isempty(projection_labels)
        projection_labels = fliplr(projection_labels);
        for n=1:numel(projection_labels)
            H5DS.set_label(ncart,n-1,projection_labels{n});
        end
    end
    
    % Close it all up
    H5D.close(ncart);
    H5S.close(space);
    H5T.close(filetype);
end



H5F.close(file);

end