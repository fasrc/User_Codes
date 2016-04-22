%===========================================================
% Program: image_processing.m
%          Function illustrates image processing in MATLAB
%===========================================================
function [] = image_process(image_name_in, image_name_out)
  im_in  = imread(image_name_in); % Read image
  im_out = histeq(im_in);         % Improve image contrast
  imwrite(im_out,image_name_out)  % Write processed image
end
