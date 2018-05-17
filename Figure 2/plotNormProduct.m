load NormProduct.mat;

subplot(1,2,1);


U2 = -5:0.01:5;

normSum = 1/sqrt(2*pi*2)*exp(-U2.*U2/4),hold on;

plot(U2,normSum);

U = -5:step:-0.001;

plot(U,DU),hold on;
plot(0-U,DU),hold on;
legend('X_1+X_2','X_1X_2')

subplot(1,2,2);

normSumD = 1/sqrt(4*pi)*exp(-U2.*U2/4).*(-U2/2),hold on;

plot(U2,abs(normSumD));
plot(U,DDU),hold on;
plot(0-U,DDU),hold on;
legend('X_1+X_2','X_1X_2')
