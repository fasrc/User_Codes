%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Program: se_fdm.m
%
% Purpose: Solutions to 1D time independent Schrodinger Equation
%          with Finite Difference Method
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [] = se_fdm( Ev, iproc )
  % +++ INPUTS +++ 
  Uv = -400;            % depth of potential well [eV]
  L  = 1e-10;           % depth of potential well [m] 
  N  = 1000;            % number of x values

  % +++ CONSTANTS +++
  hbar = 1.055e-34;      % J.s
  e    = 1.602e-19;      % C
  me   = 9.109e-31;      % kg
  
  % +++ SETUP CALCULATIONS +++
  % x coodinates
  x_min = -1*L;
  x_max = -x_min;
  x = linspace(x_min,x_max,N);
  dx = x(2)-x(1);
  
  U0 = e * Uv;              % potential well depth [J]        
  E = e* Ev;                % total energy [J]
  U = zeros(1,N);
  
  psi = zeros(1,N);         % initial conditions
  psi(2) = 1;
  
  % +++ FDM +++
  for n = 2:N-1
    if abs(x(n)) < L/2, U(n) = U0; end;
    SEconst = (2*me/hbar^2).*(E - U(n)).* dx^2;
    psi(n+1) = (2 - SEconst) * psi(n) - psi(n-1);
  end

  % Normalize wavefunction
  A = simpson1d(psi.*psi,x_min,x_max);
  psi = psi ./sqrt(A);

  % Number of crossings
  cross = 0;
  for c = 2 : N
    if psi(c-1)*psi(c) < 0, cross = cross + 1; end
  end

  fprintf('Prosess %d:\n',iproc);
  fprintf('Ev = %d\n', Ev);
  fprintf('Number of crossing for Psi = %d\n', round(cross)); 
  fprintf('End value of Psi  =  %0.3g\n', psi(end));
  disp('  ');
  disp('  ');

  % +++ FIGURES +++
  figure(iproc)
  set(gcf,'color',[1 1 1]);
  set(gcf,'Units','Normalized') 
  set(gcf,'Position',[0.3 0.2 0.6 0.4]) 
  set(gca,'fontsize',12);
  subplot(1,2,1)
  plot(x*1e9,U/e,'r','lineWidth',3)
  hold on
  plot([x(1)*1e9 x(end)*1e9],[Ev Ev],'b');
  xlabel('position  x  (nm)')
  ylabel('potential energy U (eV)');

  subplot(1,2,2)
  set(gcf,'color',[1 1 1]);
  plot(x,psi,'lineWidth',3)
  hold on
  plot([x_min x_max],[0 0],'k');
  plot([x_min/2 x_min/2],[-1.2*max(psi) max(psi)],'r','lineWidth',2);
  plot([x_max/2 x_max/2],[-1.2*max(psi) max(psi)],'r','lineWidth',2);
  plot([x_min x_min/2],[max(psi) max(psi)],'r','lineWidth',2);
  plot([x_max/2 x_max],[max(psi) max(psi)],'r','lineWidth',2);
  plot([x_min/2 x_max/2],[-1.2*max(psi) -1.2*max(psi)],'r','lineWidth',2);
  axis off
  
  % Save figure
  outfile = strcat('fig_',int2str(iproc));
  file_format = '-dpng';
  print(outfile,file_format);
  
end
