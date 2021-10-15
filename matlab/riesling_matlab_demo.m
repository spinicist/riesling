%% MATLAB Demo
% Quick demo of the basic reading and writing capabilities in the riesling
% matlab library.
%
%% --- Generate phantom data --- %%
%
% First generate a phantom dataset:
% You may have to uncomment and edit the following line to access your
% riesling binary:
setenv('PATH', '/users/tobias/Code/install/bin');
system('riesling phantom --shepp_logan sl-phantom -v');

% Read this into Matlab

[kspace, traj, info] = riesling_read('sl-phantom.h5');

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
info_us.spokes_hi = size(kspace_us, 3);

% Now we write the data back to a new file
riesling_write('sl-phantom-us.h5', kspace_us, traj_us, info_us);

%% Compare images
% Run a riesling recon of both phantom datasets to see effect of
% undersampling

system('riesling recon sl-phantom.h5 -v')
system('riesling recon sl-phantom-us.h5 -v')

image_full = riesling_read('sl-phantom-recon.h5');
image_us   = riesling_read('sl-phantom-us-recon.h5');

figure;
subplot(1,2,1);
imagesc(squeeze(abs(image_full(:,:,floor(end*0.4)))));
axis('image'); axis off; title("Fully sampled", 'fontsize', 20);

subplot(1,2,2);
imagesc(squeeze(abs(image_us(:,:,floor(end*0.4)))));
axis('image'); axis off; title("Undersampled", 'fontsize', 20)
colormap gray
