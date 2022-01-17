function [output] = riesling(cmd, K, Traj, varargin)
% RIESLING Call RIESLING command from Matlab.
%   [varargout] = RIESLING(cmd, varargin) to run given riesling command (cmd) using the
%   data arrays/matrices passed as varargin.
%
%   [A, B] = RIESLING('command', X, Y) call command with inputs X Y and outputs A B
%
%   To output a list of available riesling commands simply run "riesling". To
%   output the help for a specific riesling command type "riesling command -h".
%
% Parameters:
%   cmd:        Command to run as string (including non data parameters)
%   varargin:   Data arrays/matrices used as input
%
% Example:
%   riesling traj -h
%   recon = riesling('nufft -i traj', data) call nufft with inputs data
%

% Code heavily modified from bart.m from the bart toolbox, original by Martin
% Uecker and Soumick Chatterjee. 
% Authors:
% 2021 Patrick Fuchs (p.fuchs@ucl.ac.uk)
%

if nargin < 4
    Info = riesling_info;
else
    Info = varargin{1};
end

fileNameIn = [tempname,'.h5'];
[~,name] = fileparts(fileNameIn);

opts = strsplit(cmd);
cmd  = opts{1};
opts = sprintf('%s ',opts{2:end});

% Adjust tensor name to fit riesling convention w.r.t. nufft command (that
% is, "image" for the image domain cartesian data, and "nufft-forward" for
% the artificially generated non-uniform fourier space dataset).
if strcmpi(cmd,'nufft')
    if any(regexpi(opts,'--fwd'))
        riesling_write(fileNameIn, K, Traj, Info, 'image');
    else
        riesling_write(fileNameIn, K, Traj, Info, 'nufft-forward');
    end
else
    riesling_write(fileNameIn, K, Traj, Info, 'noncartesian');
end


try
    [ERR] = system(['riesling ',cmd,' ',fileNameIn,' ',opts]);
    
    if ERR ~= 0
        delete(fileNameIn);
        error('Failure during system call, please refer to log for more info.')
    end
    
    fileNameOut = deblank(ls([name,'*.h5']));
    output = riesling_read(fileNameOut);
    
catch 
    delete(fileNameIn);
    try
        fileNameOut = deblank(ls([name,'*.h5']));
        delete(fileNameOut);
    catch
        fprintf('No output generated.\n');
    end
end

delete(fileNameIn);
delete(fileNameOut);


end
