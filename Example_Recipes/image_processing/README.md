#### PURPOSE:

Example workflow illustrating processing of multiple images with MATLAB with
the use of job arrays in SLURM. The specific example runs 5 instances of the
MATLAB script working on 5 images at the same time. The code reads an image,
improves the image contrast, and writes out the processed image. 

#### CONTENTS:

(1) image_process.m: MATLAB image processing code.

(2) run.sbatch: Batch job submission script for sending the array job
                to the queue.

(3) images_in: Directory with input images.

(4) images_out: Directory with processed / output images.
                       
#### EXAMPLE USAGE:
	source new-modules.sh
	module load matlab/R2015b-fasrc01
	sbatch run.sbatch


#### EXAMPLE OUTPUT:

Example input image:

![Input image](images_png/image_in_2.png)

Example output image:

![Output image](images_png/image_out_2.png)

#### REFERENCES:

* [Job arrays on the FASRC cluster](https://rc.fas.harvard.edu/resources/running-jobs/#Job_arrays)

* [Image processing with MATLAB](http://www.mathworks.com/products/image)

* [Code examples](http://www.mathworks.com/products/image/code-examples.html)
