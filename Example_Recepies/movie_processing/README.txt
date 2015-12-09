PURPOSE:

Example workflow illustrating processing of multiple video files,
and reading data from CVS files. In this example the movie names 
and timecode data are read from a CSV file.

CONTENTS:

(1) movie_info.csv: File with movie names and timecode data for
                    each movie. There are 10 movies in this example.

(2) process_movies.sh: Bash script automating video processing.
                       
RUN:

The bash script is set up for a "dry run". I.e., instead of executing
the commands, it echos them to the screen. For actual production runs,
please remove the "echo" statements on lines 19 and 20 and modify commands
inside the while loop for your specific case.

Run the example workflow with

bash process_movies.sh movie_info.csv mov

EXAMPLE OUTPUT:

[pkrastev@sa01 movie_processing]$ bash process_movies.sh movie_info.csv mov
Filename: movie_01.mov
Timecode: 04:13.0
ffmpeg -ss 04:13.0 -i movie_01.mov
ffmpeg -i movie_01.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_01_10fps720p.mov

Filename: movie_02.mov
Timecode: 07:65.2
ffmpeg -ss 07:65.2 -i movie_02.mov
ffmpeg -i movie_02.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_02_10fps720p.mov

Filename: movie_03.mov
Timecode: 03:36.6
ffmpeg -ss 03:36.6 -i movie_03.mov
ffmpeg -i movie_03.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_03_10fps720p.mov

Filename: movie_04.mov
Timecode: 02:10.3
ffmpeg -ss 02:10.3 -i movie_04.mov
ffmpeg -i movie_04.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_04_10fps720p.mov

Filename: movie_05.mov
Timecode: 09:45.4
ffmpeg -ss 09:45.4 -i movie_05.mov
ffmpeg -i movie_05.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_05_10fps720p.mov

Filename: movie_06.mov
Timecode: 04:15.6
ffmpeg -ss 04:15.6 -i movie_06.mov
ffmpeg -i movie_06.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_06_10fps720p.mov

Filename: movie_07.mov
Timecode: 03:27.9
ffmpeg -ss 03:27.9 -i movie_07.mov
ffmpeg -i movie_07.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_07_10fps720p.mov

Filename: movie_08.mov
Timecode: 09:34.0
ffmpeg -ss 09:34.0 -i movie_08.mov
ffmpeg -i movie_08.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_08_10fps720p.mov

Filename: movie_09.mov
Timecode: 05:54.7
ffmpeg -ss 05:54.7 -i movie_09.mov
ffmpeg -i movie_09.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_09_10fps720p.mov

Filename: movie_10.mov
Timecode: 07:32.5
ffmpeg -ss 07:32.5 -i movie_10.mov
ffmpeg -i movie_10.mov -c:v libx264 -r 10 -s 1280x720 -b:v 5000k -threads 4 movie_10_10fps720p.mov

All DONE.
