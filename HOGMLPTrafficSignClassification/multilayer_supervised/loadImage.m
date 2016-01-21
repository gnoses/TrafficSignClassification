
clear;
index = 1;
for j = 0:42
    path = sprintf('Images/%05d',j);
    fileList = dir(path);    
    for i=3:length(fileList)
        if (length(strfind(fileList(i).name, 'ppm')) == 0)
            continue;
        end
%         str = sprintf('%s/%s',path,fileList(i).name);
%         disp(str);
%         temp = imread(str);
%         temp = rgb2gray(imresize(temp, [50 50]));
%         img(:,:,index) = temp;
        label(index) = j + 1;
        index = index + 1;        
    end
end

