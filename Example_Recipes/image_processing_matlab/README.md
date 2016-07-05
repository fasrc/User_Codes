#### PURPOSE:

Example workflow illustrating processing of multiple video files
in parallel with the help of job arrays. Each job instance works
on a separate video. The specific example loads the video files and
counts the movie frames.

#### CONTENTS:

(1) video_test.m.: MATLAB code.

(2) run.sbatch: Batch job submission script for sending the array job
                to the queue.

(3) test_mv_1.mp4  test_mv_2.mp4  test_mv_3.mp4: Input video files.
                       
#### EXAMPLE USAGE:
	source new-modules.sh
	module load matlab/R2016a-fasrc01
	sbatch run.sbatch


#### EXAMPLE OUTPUT:

```
                                                                            < M A T L A B (R) >
                                                                  Copyright 1984-2016 The MathWorks, Inc.
                                                                  R2016a (9.0.0.341360) 64-bit (glnxa64)
                                                                             February 11, 2016

 
To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.
 

	Academic License

numFrames =  1
numFrames =  2
numFrames =  3
numFrames =  4
numFrames =  5
numFrames =  6
numFrames =  7
numFrames =  8
numFrames =  9
numFrames =  10
numFrames =  11
numFrames =  12
numFrames =  13
numFrames =  14
numFrames =  15
numFrames =  16
numFrames =  17
numFrames =  18
numFrames =  19
numFrames =  20
numFrames =  21
numFrames =  22
numFrames =  23
numFrames =  24
numFrames =  25
numFrames =  26
numFrames =  27
numFrames =  28
numFrames =  29
numFrames =  30
Number of Frames:  30
```
