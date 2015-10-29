;===========================================================
; Program: pi_monte_carlo.pro
;          Computes PI via a Monte-Carlo method
;
; RUN:     idl -e pi_monte_carlo
;===========================================================
pro pi_monte_carlo
  iseed = 99
  R = 1.0d0
  darts = 1.0d6
  count = 0L
  for i = 1, darts do begin
     x = randomu(iseed,1)
     y = randomu(iseed,1)
     if ( (x^2 + y^2) LE R^2 ) then begin
        count = count + 1
     endif
  endfor
  myPI = 4.0d0 * double(count) / double(darts)
  print,"Computed PI:", myPI
  print,"Exact PI:", !PI
end
