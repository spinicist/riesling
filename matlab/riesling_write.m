function [] = riesling_write(fname, data, traj, info)
% WRITE_RIESLING writes radial k-space data to riesling .h5 format
%
% Input:
%   - fname: output filename
%   - data: Complex kspace data [nrcv, npoints, nspokes, nvol]
%   - traj: Trajectory [3, npoints, nspokes]
%   - info: Info struct
%
% Inspired by: https://stackoverflow.com/questions/46203309/write-complex-numbers-in-an-hdf5-dataset-with-matlab
% Emil Ljungberg, King's College London, 2021


if isfile(fname)
    error("%s already exists. Please delete or choose a different output name\n",fname);
end

check_fields = fieldnames(riesling_info());
info_fields = fieldnames(info);
if ~isequal(intersect(check_fields, info_fields, 'stable'), check_fields)
    error("Header fields are incorrect. Use riesling_info to generate template header");
end

fprintf('Opening %s\n',fname);
file = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Separating complex data
wdata = struct;
wdata.r = real(data);
wdata.i = imag(data);

floatType = H5T.copy('H5T_NATIVE_FLOAT');
sz = [H5T.get_size(floatType), H5T.get_size(floatType)];

% Computer the offsets to each field. The first offset is always zero.
offset = [0, sz(1)];

% Create the compound datatype for the file and for the memory (same).
filetype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert(filetype, 'r', offset(1), floatType);
H5T.insert(filetype, 'i', offset(2), floatType);

% Ensure the data-space is 4D
dims = ones(1, 4);
dims(1:ndims(data)) = size(data);
dims = fliplr(dims);
space = H5S.create_simple(4, dims, []);
ncart = H5D.create(file, 'noncartesian', filetype, space, 'H5P_DEFAULT');
H5D.write(ncart, filetype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', wdata);
H5D.close(ncart);

% Close it all up
H5S.close(space);
H5T.close(filetype);
H5F.close(file);

% To make life a bit easier we use the high-level functions to save the rest
h5create(fname, '/trajectory', size(traj));
h5write(fname, '/trajectory', traj);

% Using old hdf5write function to write compound data not supported by the
% newer h5write
hdf5write(fname, '/info', info, 'WriteMode', 'append');

disp("Closing file");

end