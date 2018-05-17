s1 = 1;
s2 = 1;

step = 0.001;

U = -5:step:-0.001;

U = U((U~=0));

DU = zeros(1,length(U));

DDU = zeros(1,length(U));

for i = 1:length(U)
    DU(1,i) = Normal_Product_Distribution(U(1,i)+(step/2),s1,s2);
end

for i = 2:length(U)
    DDU(1,i) = (DU(1,i)-DU(1,i-1))/step;
end

temp = 0;
for i = 2:length(U)
    temp = temp + DDU(1,i)^2*step;
end


