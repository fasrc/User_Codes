%===========================================================
% Program: multi_fig.m
%          Illustrates multi-figures in MATLAB
%
% Run: matlab -nosplash -nodesktop -nodisplay -r "multi_fig"
%===========================================================
% Generate some data
x = 0:0.1:10;
y1 = sin(x);
y2 = cos(x);
y3 = sin(x)+cos(x);
y4 = sin(2*x);
y5 = cos(2*x);
y6 = sin(2*x)+cos(2*x);
y7 = sin(3*x);
y8 = cos(3*x);
y9 = sin(3*x)+cos(3*x);

% Plot 1 on a 3 X 3 grid
figure
subplot(3,3,1)
plot(x,y1)
title('Subplot 1')

% Plot 2
subplot(3,3,2)
plot(x,y2)
title('Subplot 2')

% Plot 3
subplot(3,3,3)
plot(x,y3)
title('Subplot 3')

% Plot 4
subplot(3,3,4)
plot(x,y4)
title('Subplot 4')

% Plot 5
subplot(3,3,5)
plot(x,y5)
title('Subplot 5')

% Plot 6
subplot(3,3,6)
plot(x,y6)
title('Subplot 6')

% Plot 7
subplot(3,3,7)
plot(x,y7)
title('Subplot 7')

% Plot 8
subplot(3,3,8)
plot(x,y8)
title('Subplot 8')

% Plot 9
subplot(3,3,9)
plot(x,y9)
title('Subplot 9')

% Set figure size on inches and print it out
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 5 3];
fig.PaperPositionMode = 'manual';

print('figure','-dtiff')

exit
