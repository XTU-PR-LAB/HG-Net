 % Reads the list of frames.
frameList = get_synched_frames(sceneDir);

 % Displays each pair of synchronized RGB and Depth frames.
for ii = 1 : 1 : numel(frameList)
  imgRgb = imread([sceneDir '/' frameList(ii).rawRgbFilename]);
  imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(ii).rawDepthFilename]));

  figure(1);
  % Show the RGB image.
  subplot(1,3,1);
  imagesc(imgRgb);
  axis off;
  axis equal;
  title('RGB');

  % Show the Raw Depth image.
  subplot(1,3,2);
  imagesc(imgDepthRaw);
  axis off;
  axis equal;
  title('Raw Depth');
  caxis([800 1100]);

  % Show the projected depth image.
  imgDepthProj = get_projected_depth(imgDepthRaw);
  subplot(1,3,3);
  imagesc(imgDepthProj);
  axis off;
  axis equal;
  title('Projected Depth');

  pause(0.01);
end