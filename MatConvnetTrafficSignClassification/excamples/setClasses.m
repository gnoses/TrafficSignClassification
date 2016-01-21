clear;
load('trafficSign.mat');
imdb.images.data = single(imdb.images.data);
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:62,'uniformoutput',false)


save('trafficSign.mat','imdb');
clear;
load('trafficSign.mat');