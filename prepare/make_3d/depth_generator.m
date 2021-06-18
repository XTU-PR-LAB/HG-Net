dump_path = 'E:\datasets\make_3d\format\train\depths';
depth_prefix = 'E:\datasets\make_3d\train\depths\Train400Depth';
depth_files = dir(fullfile(depth_prefix,'\*.mat'));

for i = 1 : 400
    depth_path = fullfile(depth_prefix,depth_files(i).name);
    p = load(depth_path);
    depth = p.Position3DGrid(:,:,4);
    %max_depth = 80; %max(max(depth));
    %depth(depth < 1e-3) = 1e-3;
    %depth(depth > 80) = 80;
    depth_resized = imresize(depth, [640,480], 'nearest');
    depth_resized = rot90(depth_resized,3);
    depth_resized = fliplr(depth_resized);
    fip = fopen(fullfile(dump_path,sprintf('%010d.bin',i-1)),'w');
    fwrite(fip,depth_resized,'float');
    fclose(fip);
    fprintf('image processed %s\n',depth_files(i).name);
end
    