function [] = write_riesling(fname, data, traj, info)
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

fprintf('Opening %s\n',fname);
file = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Separating complex data
wdata = struct;
wdata.r = real(data);
wdata.i = imag(data);

doubleType = H5T.copy('H5T_NATIVE_DOUBLE');
sz = [H5T.get_size(doubleType), H5T.get_size(doubleType)];

% Computer the offsets to each field. The first offset is always zero.
offset = [0, sz(1)];

% Create the compound datatype for the file and for the memory (same).
filetype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert(filetype, 'r', offset(1), doubleType);
H5T.insert(filetype, 'i', offset(2), doubleType);

space = H5S.create_simple(ndims(data), fliplr(size(data)), []);
grp = H5G.create(file, 'volumes','H5P_DEFAULT','H5P_DEFAULT','H5P_DEFAULT');

for i=1:size(data,4)
    volstr = sprintf("%04d",i-1);
    fprintf("Writing volume %s\n", volstr);
    dset = H5D.create (grp, volstr, filetype, space, 'H5P_DEFAULT');
    H5D.write (dset, filetype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', wdata);
    H5D.close(dset);
end

% Close it all up
H5G.close(grp);
H5S.close(space);
H5T.close(filetype);
H5F.close(file);

% To make life a bit easier we use the high-level functions to save the rest
h5create(fname, '/traj', size(traj));
h5write(fname, '/traj', traj);

% Using old hdf5write function to write compound data not supported by the
% newer h5write
hdf5write(fname, '/info', info, 'WriteMode', 'append');         %#ok<HDFW>

disp("Closing file");

end