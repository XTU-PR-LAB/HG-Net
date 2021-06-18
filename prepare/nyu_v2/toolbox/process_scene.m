function process_scene(sceneDir,dump_path)
    frameList = get_synched_frames(sceneDir);
    for ii = 1 : 3 : numel(frameList)
      if (~isfield(frameList,'rawRgbFilename')) || (~isfield(frameList,'rawDepthFilename'))
          disp(sprintf('image or depth %d lost\n',ii));
          continue
      end
      imgRgb = imread([sceneDir '/' frameList(ii).rawRgbFilename]);
      imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(ii).rawDepthFilename]));
      imgDepthProj = get_projected_depth(imgDepthRaw);
      imgDepthProjRotate = rot90(imgDepthProj,1);
      imgDepthProjRotate = flipud(imgDepthProjRotate);
      scene_split = strsplit(sceneDir,'/');
      scene_name = scene_split{end};
      imwrite(imgRgb,[dump_path '/image/' scene_name '_' num2str(ii, '%010d') '.png']);
      %fip = fopen([dump_path '/depth/' scene_name '_' num2str(ii, '%010d') '.bin'],'wb');
      %fwrite(fip,imgDepthProjRimwrite(imgRgb,[dump_path '/image/' scene_name '_' num2str(ii, '%010d') '.png']));
      fip = fopen([dump_path '/depth/' scene_name '_' num2str(ii, '%010d') '.bin'],'wb');
      fwrite(fip,imgDepthProjRotate,'float');
      fclose(fip);
      %imshow(imgDepthProjRotate);
    end
end