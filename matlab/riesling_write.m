function [] = riesling_write(fname, data, traj, matrix, info, dim_labels)
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
% David Leitao, King's College London, 2024

if isfile(fname)
    error("%s already exists. Please delete or choose a different output name\n", fname);
end

fprintf('Opening %s\n',fname);
file = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Separate complex data
wdata = struct;
wdata.r = real(data);
wdata.i = imag(data);
floatType = H5T.copy('H5T_NATIVE_FLOAT');
sz = [H5T.get_size(floatType), H5T.get_size(floatType)];
% Compute the offsets to each field. The first offset is always zero.
offset = [0, sz(1)];

% Create the compound datatype for the file and for the memory (same).
filetype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert(filetype, 'r', offset(1), floatType);
H5T.insert(filetype, 'i', offset(2), floatType);

Ndims = 5;
if ndims(data) == 5
    dims = size(data);
else
    dims = ones(1, 5);
    dims(1:ndims(data)) = size(data);
end
dims = fliplr(dims);

space = H5S.create_simple(Ndims, dims, []);
ncart = H5D.create(file, '/data', filetype, space, 'H5P_DEFAULT');
H5D.write(ncart, filetype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', wdata);

if nargin == 6
    dim_labels = fliplr(dim_labels);
    for n=1:numel(dim_labels)
        H5DS.set_label(ncart,n-1,dim_labels{n});
    end
end

% Close it all up
H5D.close(ncart);
H5S.close(space);
H5T.close(filetype);
H5F.close(file);

% Add matrix attribute to trajectory dataset (if supplied within info)
if nargin > 3 && ~isempty(traj)
    % To make life a bit easier we use the high-level functions to save the rest
    h5create(fname, '/trajectory', size(traj));
    h5write(fname, '/trajectory', traj);

    % Reopen the file and the dataset
    file_id = H5F.open(fname, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    dataset_id = H5D.open(file_id, '/trajectory');
    attr_space_id = H5S.create('H5S_SCALAR');
    % Define the type as an array of 3 integers using H5T_ARRAY
    attrValue = int32(matrix(:)'); % Integer array with 3 elements
    base_type_id = H5T.copy('H5T_NATIVE_INT');  % Base type is integer
    attr_type_id = H5T.array_create(base_type_id, 1, numel(attrValue));  % Create a 1D array of size 3
    % Create the attribute
    attr_id = H5A.create(dataset_id, 'matrix', attr_type_id, attr_space_id, 'H5P_DEFAULT');
    % Write the attribute data
    H5A.write(attr_id, attr_type_id, attrValue);
    % Close the attribute, dataset, and file
    H5A.close(attr_id);
    H5T.close(attr_type_id);
    H5S.close(attr_space_id);
    H5D.close(dataset_id);
    H5F.close(file_id);
end

if nargin < 5
    info = riesling_info();
else
    check_fields = fieldnames(riesling_info());
    info_fields = fieldnames(info);
    if ~isequal(intersect(check_fields, info_fields, 'stable'), check_fields)
        error("Header fields are incorrect. Use riesling_info to generate template header");
    end
    % Check info data sizes
    ref_info = riesling_info();
    for i=1:length(check_fields)
        size_in = size(info.(check_fields{i}));
        size_ref = size(ref_info.(check_fields{i}));
        if ~isequal(size_in, size_ref)
            error("info.%s is the wrong size. Currently (%d,%d), should be (%d,%d)",... 
                check_fields{i}, size_in(1), size_in(2), size_ref(1), size_ref(2));
        end
    end
end
hdf5write(fname, '/info', info, 'WriteMode', 'append'); % Compound data not supported by the newer h5write

end