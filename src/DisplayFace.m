%---------------------------------------------------
% author    : SHI XUDONG
% title     : Age Face Recognition main code
% date      : 2014.04.01
%---------------------------------------------------

function [] = DisplayFace(vector)
% display vector as face (pixel value 0-255)
% input:
% vector     :   vecotr to be displayed

rows = 64;
cols = 64;

% rescale gray value to [0 255]
vector = double(vector);
min_v = min(vector);
max_v = max(vector);
vec_norm = (vector - min_v)./(max_v - min_v);
vector_0_255 = uint8(255*vec_norm);

% display
face = reshape(vector_0_255, rows, cols);
imshow(face);

end