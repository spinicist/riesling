function info = riesling_info()

info.type = 1;                  % 1 = 3D trajectory, 2 = stack-of-stars type
info.channels = 1;              % Number of channels in receive coil
info.matrix = ones(3,1);        % Nominal matrix size
info.read_points = 1;           % Read-out points per spoke
info.spokes = 1;                % Number of spokes in acquisition
info.volumes = 1;               % Number of volumes in sequential acquisition
info.frames = 1;                % Number of frames/echoes/temporal points per volume
info.tr = 1.0;                  % TR, should be milliseconds
info.voxel_size = ones(3,1);    % Nominal voxel-size, should be in mm
info.origin = zeros(3,1);    % Physical space co-ordinate of 0,0,0 voxel (ITK convention)
info.direction = reshape(eye(1,9), [9,1]); % Direction matrix (ITK convention)
