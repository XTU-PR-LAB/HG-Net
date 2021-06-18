datasetDir = '/media/franc_zi/Elements SE/datasets/nyu/nyu_depth/clips';
dump_path = '/media/franc_zi/Elements SE/datasets/nyu/nyu_depth/nyu_processed';
scenes = regexp(ls(datasetDir), '(\s+|\n)', 'split');
scenes(end) = [];
scenes = sort(scenes);
for ii = 1 : 1 : numel(scenes)
    disp(['scenes processed ' num2str(ii) '/' num2str(numel(scenes))])
    dir = sprintf('%s/%s', datasetDir, scenes{ii});
    process_scene(dir, dump_path);
end
