function bessel_value = Modified_Bessel_Zero(x)

% step = 0.001;
% T = step:step:100 + step/2;
% bessel_value = sum(cos(x*T)./sqrt(T.*T+1)*step);

%bessel_value = quad(@(t)(cos(x*t)./sqrt(1+t.^2)),0,100);

bessel_value = besselk(0,x);

end