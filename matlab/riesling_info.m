function info = riesling_info()

info.voxel_size = ones(3, 1); % Nominal voxel-size, should be in mm
info.origin = zeros(3, 1);    % Physical space co-ordinate of 0,0,0 voxel (ITK convention)
info.direction = eye(3, 3);   % Direction matrix (ITK convention)
info.tr = 1.0;                % TR, should be milliseconds
