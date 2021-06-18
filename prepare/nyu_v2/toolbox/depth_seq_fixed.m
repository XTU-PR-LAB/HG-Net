dump_path = '/media/franc_zi/Elements SE/datasets/nyu/nyu_depth/nyu_processed/fix_depth'; %/media/franc_zi/Elements SE/datasets/nyu/nyu_depth/nyu_processed/fix_depth
depth_prefix = '/media/franc_zi/Elements SE/datasets/nyu/nyu_depth/nyu_processed/depth'; %/media/franc_zi/Elements SE/datasets/nyu/nyu_depth/nyu_processed/depth
image_prefix = '/media/franc_zi/Elements SE/datasets/nyu/nyu_depth/nyu_processed/image'; %/media/franc_zi/Elements SE/datasets/nyu/nyu_depth/nyu_processed/image
depth_files = dir(fullfile(depth_prefix,'*.bin'));
img_files = dir(fullfile(image_prefix,'*.png'));%45151
%disp(img_files);
t1 = clock;
for i = 1 : 45151
    img_path = fullfile(image_prefix,img_files(i).name);
    depth_path = fullfile(depth_prefix,depth_files(i).name);
    imgRgb = imread(img_path);
    %imshow(imgRgb);
    %imgRgb = imresize(imgRaw,[256,512],'bilinear');
    fip = fopen(depth_path);
    imgDepthAbs = fread(fip,[640,480],'float');
    imgDepthAbs = rot90(imgDepthAbs,3);
    imgDepthAbs = fliplr(imgDepthAbs);
    %imshow(imgDepthAbs);
    imgDepthFilled = fill_depth_colorization(imgRgb,imgDepthAbs);
    fclose(fip);
    imgDepthFilled = rot90(imgDepthFilled,3);
    imgDepthFilled = fliplr(imgDepthFilled);
    fip = fopen(fullfile(dump_path,depth_files(i).name),'w');    
    fwrite(fip,imgDepthFilled,'float');
    fclose(fip);
    if mod(i,100) == 0
        t2 = clock;
        fprintf('cur_idx: %d avg time count per 100 images: %.4f s\n', i, etime(t2,t1)/100);
        t1 = clock;
    end
    %colormap(jet);
    %figure(1);
    
    %imagesc(imgRgb);
    %subplot(1,3,2); imagesc(imgDepthAbs);
    %subplot(1,3,3);
    
    %imagesc(imgDepthFilled);
    %fprintf('image %d processed %s time_count: %.4fs\n', i, depth_files(i).name, etime(t2,t1));
end
    