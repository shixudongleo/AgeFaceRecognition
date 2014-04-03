%---------------------------------------------------
% author    : SHI XUDONG
% title     : Age Face Recognition main code
% date      : 2014.04.01
%---------------------------------------------------

function [X Y] = ReadData(load_list)
% load data
% input:
% load_list     :   strings with file names to be loaded
%
% ouput:
% X             :   load faces into data matrix
% Y             :   classification label

face_folder = '../data/AgeFaceDataset/';
data_path = strrep(load_list, '../data/', face_folder);
data_path = strrep(data_path, '.txt', '.mat');

if exist(data_path)
    load(data_path);
    return
end

fid = fopen(load_list, 'r');
if (fid == -1)
    error('can not read %s\n', load_list);
end

cell_list = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);

list_len = length(cell_list{1});
img1 = imread([face_folder, cell_list{1}{1}]);
[rows cols] = size(img1);
X = zeros(list_len, rows*cols, 'uint8');
Y = zeros(list_len, 1);

for ii = 1:list_len
    tmp_img = imread([face_folder, cell_list{1}{ii}]);
    X(ii, :) = tmp_img(:);
    Y(ii) = str2num(cell_list{1}{ii}(1:3));
end

X = double(X);
Y = int8(Y);
save(data_path, 'X', 'Y');

end