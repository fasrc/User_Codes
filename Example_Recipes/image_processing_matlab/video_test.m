%====================================================================
% Program: video_test.m
% Load a video file and count number of movie frames
%====================================================================
function [] = video_test(mov_name)

% Construct a VideoReader object
mov = VideoReader(mov_name);

% Read one frame at a time using readFrame until the end of the file is reached. 
% Append data from each video frame to the structure array.
numFrames = 0;
while hasFrame(mov)
  numFrames = numFrames + 1;
  fprintf('numFrames =  %d\n', numFrames);
  s(numFrames).cdata = readFrame(mov);
end

% Print out number offrames
fprintf('Number of Frames:  %d\n', numFrames);

end
