%% MATLAB Demo
% Quick demo of the basic reading and writing capabilities in the riesling
% matlab library.
%
% Emil Ljungberg, King's College London
%
%% --- Generate phantom data --- %%
%
% First generate a phantom dataset in your terminal:
%
% $ rielsing phantom --shepp_logan sl_phantom
% 
% This will greate a phantom dataset which we can run our tests with

[kspace, traj, info] = read_riesling('sl_phantom.h5');

% Size of datastructures
% kspace: [nrcv, npoints, nspokes, nvol]
% traj: [3, npoints, nspokes]

[nrcv, npoints, nspokes] = size(kspace);

disp('Information in info structure')
disp(info)

%% Visualise trajectory and k-space data
figure()
labels = ["Gx","Gy","Gz"];
for i=1:3
   subplot(3,1,i);
   plot(1:nspokes,squeeze(traj(i,end,:)));
   xlabel('Spoke number'); ylabel(labels{i});
   axis([1,nspokes,-1,1]);
   grid on
end

figure()
imagesc(squeeze(log(abs(kspace(1,:,:))))); 
xlabel('Spoke number');
ylabel('Readout point');
colormap gray
title('K-space data first coil');

%% Modify data and write back
% Now we will subsample the data and write it back to the .h5 file
% We pick every second spoke in the data and trajectory

kspace_us = kspace(:,:,1:2:end);
traj_us = traj(:,:,1:2:end);

% We also need to modify the infor structure
info_us = info;
info_us.spokes_hi = size(kspace_us,3);

% Now we write the data back to a new file
write_riesling('sl_phantom_us.h5', kspace_us, traj_us, info_us);

%% Compare images
% Run a riesling recon of both phantom datasets to see effect of
% undersampling
%
% $ riesling rss sl_phantom.h5
% $ riesling rss sl_phantom_us.h5
%
% View it using your favourite nifti viewer
% $ fsleyes sl_phantom-rss.nii sl_phantom_us-rss.nii
