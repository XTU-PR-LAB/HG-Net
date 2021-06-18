%datasetDir = '/media/lovpodtt/EAGET忆捷/nyu_depth/clips';
%scenes = regexp(ls(datasetDir), '(\s+|\n)', 'split');
%scenes(end) = [];

%dir = sprintf('%s/%s', datasetDir, scenes{1})
%scene_name = strsplit(dir,'/');
%str2 = scene_name{end}

a = ['image' '/ok/' 'hello']

