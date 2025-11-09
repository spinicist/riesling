function [output] = riesling(cmd, varargin)
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
% 2025 David Leitao (david.leitao@kcl.ac.uk)

% Default options:
keepInput = false;
keepOutput = false;
f0Map = [];
t2Map = [];
for n=1:numel(varargin)
    if strcmpi(varargin{n},'data')
        data = varargin{n+1};
    end
    if strcmpi(varargin{n},'traj')
        traj = varargin{n+1};
    end
    if strcmpi(varargin{n},'matrix')
        matrix = varargin{n+1};
    end
    if strcmpi(varargin{n},'info')
        info = varargin{n+1};
    end
    if strcmpi(varargin{n},'input')
        keepInput = true;
        fileNameIn = varargin{n+1};
    end
    if strcmpi(varargin{n},'output')
        keepOutput = true;
        fileNameOut = varargin{n+1};
    end
    if strcmpi(varargin{n},'f0map')
        f0Map = varargin{n+1};
    end
    if strcmpi(varargin{n},'t2map')
        t2Map = varargin{n+1};
    end
end

if ~keepInput
    fileNameIn = [tempname,'.h5'];
else
    % %check if fileNameIn already exists, if so delete to replace
    % if exist(fileNameIn, 'file')
    %     delete(fileNameIn);
    % end
end
% [~,name] = fileparts(fileNameIn);

opts = strsplit(cmd);
cmd  = opts{1};
opts = sprintf('%s ',opts{2:end});

% Adjust tensor name to fit riesling convention w.r.t. nufft command (that
% is, "image" for the image domain cartesian data, and "nufft-forward" for
% the artificially generated non-uniform fourier space dataset).
switch cmd
    case 'nufft'
        if any(regexpi(opts,'--fwd'))
            riesling_write(fileNameIn, data, traj, info, 'image');
        else
            riesling_write(fileNameIn, data, traj, info, 'nufft-forward');
        end
    case {'sense-maps','psf','denoise'}
        if ~keepInput
            fileNameIn = [];
        end

    otherwise
        riesling_write(fileNameIn, data, traj, matrix, info, [], f0Map, t2Map);
end


try
    if ~keepOutput
        fileNameOut = [tempname,'.h5'];
    end

    [ERR] = system(['riesling ',cmd,' ',opts,' ',fileNameIn,' ',fileNameOut]);
    
    if ERR ~= 0 && ~keepInput
        delete(fileNameIn);
        error('Failure during system call, please refer to log for more info.')
    end
    
    % if keepOutput
        output = [];
    % else
        % output = riesling_read(fileNameOut);
    % end
    
catch 
    if ~keepInput
        delete(fileNameIn);
    end
    try
        % fileNameOut = deblank(ls([name,'*.h5']));
        delete(fileNameOut);
    catch
        fprintf('No output generated.\n');
    end
end
if ~keepInput
    delete(fileNameIn);
end
if ~keepOutput
    delete(fileNameOut);
end


end
