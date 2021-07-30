function info = riesling_info()

info.type = 1;         % 1 = 3D trajectory, 2 = stack-of-stars type
info.channels = 1;     % Number of channels in receive coil
info.matrix = [1 1 1]; % Nominal matrix size
info.read_points = 1;  % Read-out points per spoke
info.read_gap = 0;     % ZTE dead-time gap
info.spokes_hi = 1;    % Number of spokes in main acquisition
info.spokes_lo = 0;    % Number of spokes in a ZTE WASPI-type acquisition
info.lo_scale = 0;     % Scaling factor for WASPI-type acquisition
info.volumes = 1;      % Number of volumes in sequential acquisition
info.echoes = 1;       % Number of echoes
info.tr = 1;           % TR, should be milliseconds
info.voxel_size = [1 1 1]; % Nominal voxel-size, should be in mm
info.origin = [0 0 0]; % Physical space co-ordinate of 0,0,0 voxel (ITK convention)
info.direction = ones(3); % Direction matrix (ITK convention)
