% SAN code
function [flag,loss,Model,f_temp] = SANs(N,label,Niter,net,errorThreshold,delta,lr,fig,Model)
% close all;
%% Learing parameterss
% errorThreshold = 0.02;
% delta.start = 0.1;
% delta.rate = 0.9;
% delta.Step = 100;
% lr.start = 0.05;
% lr.rate = 0.9;
% lr.Step = 100;

%% Sample data
data = ones(1,N);
flag = 0;
type = length(label);
%% Active function
funAct = @(x)max(0,x); %RealU
% funAct = @(x)(1 ./ (1+exp(-x))); %Sigmoid
% funAct = @(x)(x);

%% SoftMax
% funSoft = @(x)(1 ./ (1+exp(-x))); %Sigmoid
funSoft = @(x)(x);
Label = [];
for i = 1:type
    label{i} = funSoft(label{i});
    Label = [Label label{i}];
end
mean_label = mean(Label);
%% Initialize
L = length(net);
d = size(data,1); M = size(data,2);
if isempty(Model)
    W = cell(1,L);B = cell(1,L);
    C = cell(1,type);D = cell(1,type);
    Rmu = fminsearch(@(x)abs(funmuoutput(type,net,funAct,funSoft,x,L)-mean_label), 0.5);
    Rsigma = 0.0001;
    for i = 1:L
        if i == 1
            W{i} = {Rmu*ones(net(i),d),Rsigma*ones(net(i),d)};
            B{i} = {Rmu*ones(net(i),1),Rsigma*ones(net(i),1)};
        else
            W{i} = {Rmu*ones(net(i),net(i-1)),Rsigma*ones(net(i),net(i-1))};
            B{i} = {Rmu*ones(net(i),1),Rsigma*ones(net(i),1)};
        end
    end
    for i = 1:type
        C{i} = {ones(1,net(L)),Rsigma*ones(1,net(L))};
        D{i} = [1,Rsigma*1];
    end
else
    W = Model.W;
    B = Model.B;
    C = Model.C;
    D = Model.D;
end
f = funOutput(type,net,funAct,funSoft,W,B,C,D,L,data,M);
[~, f_errorS] = funError(type, f, label);
fprintf('### Error of Inilization or model: %.3e.\n', f_errorS);

%% Initialize Temp
C_temp = C;
D_temp = D;

%% Learning Process
loss = zeros(1,Niter);
which_layer = L;
i_last = 0;
for i = 1:Niter
    f = funOutput(type,net,funAct,funSoft,W,B,C,D,L,data,M);
    [~,f_error] = funError(type, f, label);
    if mod(i,5) == 0, fprintf('### Step %d, layer = %d, lr = %.3e, delta = %.3e, error = %.3e.\n',...
            i, which_layer, studyRate, delta_der, f_error);end
    loss(i) = f_error; f_temp = f;
    delta_der = delta.start*delta.rate^(i/delta.Step);
    studyRate = lr.start*lr.rate^(i/lr.Step);
    if f_error < errorThreshold
        flag = 1;break;
    end
    
    %% Switch learning layer
    if i >= i_last+50 && (loss(i) > mean(loss(i-10:i-1)))
        i_last = i;
        if which_layer == 1
            which_layer = L;
        else
            which_layer = which_layer-1;
        end
    end
    
    %% Learning start from the last layer
    for j = which_layer
        [W_delta,B_delta,C_delta,D_delta] = funInitialize_delta(type,net,d,L);
        if j == 1, W_row = net(1,j); W_column = d;
        else W_row = net(1,j); W_column = net(1,j-1);end
        B_row = net(1,j); B_column = 1;
        
        for w_k = 1:2 %%learn W
            for k1 = 1:W_row
                for k2 = 1:W_column
                    W_temp = W;
                    W_temp{j}{w_k}(k1,k2) = W{j}{w_k}(k1,k2) + delta_der;
                    f_temp = funOutput(type,net,funAct,funSoft,W_temp,B,C,D,L,data,M);
                    [f_errortemp_sep,f_errortemp] = funError(type, f_temp, label);
                    W_delta{j}{w_k}(k1,k2) = (-studyRate) * (f_errortemp - f_error) / delta_der;
                end
            end
        end
        
        for w_k = 1:2 %%learn B
            for k1 = 1:B_row
                for k2 = 1:B_column
                    B_temp = B;
                    B_temp{j}{w_k}(k1,k2) = B{j}{w_k}(k1,k2) + delta_der;
                    f_temp = funOutput(type,net,funAct,funSoft,W,B_temp,C,D,L,data,M);
                    [f_errortemp_sep,f_errortemp] = funError(type, f_temp, label);
                    B_delta{j}{w_k}(k1,k2) = (-studyRate) * (f_errortemp - f_error) / delta_der;
                end
            end
        end
        [W,B,C,D] = addSteps(type,W,B,C,D,W_delta,B_delta,C_delta,D_delta,L);
    end
    
    %% Learning the output layer
    [W_delta,B_delta,C_delta,D_delta] = funInitialize_delta(type,net,d,L);
    f = funOutput(type,net,funAct,funSoft,W,B,C,D,L,data,M);
    [f_error_sep,f_error] = funError(type,f, label);
    C_row = 1; C_column = net(L);
    for c_cell = 1:type
        for w_k = 1:2 %%learn C
            for k1 = 1:C_row
                for k2 = 1:C_column
                    C_temp{c_cell} = C{c_cell};
                    C_temp{c_cell}{w_k}(k1,k2) = C{c_cell}{w_k}(k1,k2) + delta_der;
                    f_temp = funOutput(type,net,funAct,funSoft,W,B,C_temp,D,L,data,M);
                    [f_errortemp_sep,f_errortemp] = funError(type, f_temp, label);
                    C_delta{c_cell}{w_k}(k1,k2) = (-studyRate) * (f_errortemp_sep(c_cell) - f_error_sep(c_cell)) / delta_der;
                end
            end
        end
    end
    for d_cell = 1:type
        for w_k = 1:2 %%learn D
            D_temp{d_cell} = D{d_cell};
            D_temp{d_cell}(1,w_k) = D{d_cell}(1,w_k) + delta_der;
            f_temp = funOutput(type,net,funAct,funSoft,W,B,C,D_temp,L,data,M);
            [f_errortemp_sep,f_errortemp] = funError(type, f_temp, label);
            D_delta{d_cell}(1,w_k) = (-studyRate) * (f_errortemp_sep(d_cell) - f_error_sep(d_cell)) / delta_der;
        end
    end
    
    %% Learning with step
    [W,B,C,D] = addSteps(type,W,B,C,D,W_delta,B_delta,C_delta,D_delta,L);
end
Model.W = W;
Model.B = B;
Model.C = C;
Model.D = D;

save label.mat label;
save fOutput.mat f_temp;
save loss.mat loss;
% fprintf('Total Iteration NO. : %d with error %.3e.\n', i, f_error);
loss = loss(1:i);
if fig == 1
    figure;plot(1:i,loss(1:i));
    figure;
    subplot(1,2,1);
    hist(cell2mat(f_temp),40);
    hold on;
    subplot(1,2,2);
    hist(cell2mat(label),40);
%     outputW(type,W,B,C,D,L);
end
end

%% Initilize the delta
function [W_delta,B_delta,C_delta,D_delta] = funInitialize_delta(type,net,d,L)
for i = 1:L
    if i == 1
        W_delta{i} = {zeros(net(i),d),zeros(net(i),d)};
    else
        W_delta{i} = {zeros(net(i),net(i-1)),zeros(net(i),net(i-1))};
    end
    B_delta{i} = {zeros(net(i),1),zeros(net(i),1)};
end
for cd_cell = 1:type
    C_delta{cd_cell} = {zeros(1,net(L)),zeros(1,net(L))};
    D_delta{cd_cell} = [0,0];
end
end

%% Compute output of NN
function f_sep = funOutput(type, net,funAct,funSoft,W,B,C,D,L,data,M)
x = cell(1,L+1);
x{1} = data;
d = size(data,1);
for i = 1:L
    x{i+1} = zeros(net(1,i),M);
    if i == 1, xl = d;
    else xl = net(1,i-1);
    end
    for m = 1:net(1,i)
        for l = 1:xl
            x{i+1}(m,:) = x{i+1}(m,:) + normrnd(W{i}{1}(m,l),W{i}{2}(m,l),1,M) .* repmat(x{i}(l),1,M);
        end
    end
    x{i+1} = funAct(x{i+1}+normrnd(B{i}{1}(m,1),B{i}{2}(m,1),net(1,i),M));
end
f_sep = cell(1,type);
for i = 1:type
    f_sep{i} = zeros(1,M);
end
for c_cell = 1:type
    for m = 1:net(1,L)
        f_sep{c_cell} = f_sep{c_cell} + normrnd(C{c_cell}{1}(1,m),C{c_cell}{2}(1,m),1,M) .* x{L+1}(m,:);
    end
end
for d_cell = 1:type
    f_sep{d_cell} = funSoft(f_sep{d_cell} + normrnd(D{d_cell}(1),D{d_cell}(2),1,M));
end
end

%% Compute error of NN
function [ferror_sep, ferror] = funError(type,f1,f2)
global norm_type
band_num = 100;
ferror_sep = zeros(1,type);
for i = 1:type
    lower_boundary = min(min(f1{i}),min(f2{i}));
    upper_boundary = max(max(f1{i}),max(f2{i}));
    hist_f1=hist([f1{i},lower_boundary,upper_boundary],band_num) / length(f1{i});
    hist_f2=hist([f2{i},lower_boundary,upper_boundary],band_num) / length(f2{i});
%         ferror_sep(i)=sum(abs(hist_f1-hist_f2));
    ferror_sep(i)=norm(hist_f1-hist_f2,norm_type);
end
ferror = sum(ferror_sep);
% if type > 1
%     % ferror = sum(ferror_sep);
%     f1_total = cell2mat(f1);
%     f2_total = cell2mat(f2);
%     lower_boundary = min(min(f1_total),min(f2_total));
%     upper_boundary = max(max(f1_total),max(f2_total));
%     hist_f1=hist([f1_total,lower_boundary,upper_boundary],band_num) / length(f1_total);
%     hist_f2=hist([f2_total,lower_boundary,upper_boundary],band_num) / length(f2_total);
%     % ferror=sum(abs(hist_f1-hist_f2));
%     ferror=norm(hist_f1-hist_f2);
% else
%     ferror=ferror_sep;
% end
end

%% Add steps
function [W,B,C,D] = addSteps(type,W,B,C,D,W_delta,B_delta,C_delta,D_delta,L)

for i = 1:L
    W{i}{1} = W{i}{1} + W_delta{i}{1};
    B{i}{1} = B{i}{1} + B_delta{i}{1};
end

for i = 1:type
    C{i}{1} = C{i}{1} + C_delta{i}{1}; D{i}(1,1) = D{i}(1,1) + D_delta{i}(1,1);
    C{i}{2} = C{i}{2} + C_delta{i}{2}; D{i}(1,2) = D{i}(1,2) + D_delta{i}(1,2);
end

for i = 1:L
    W{i}{2} = W{i}{2} + W_delta{i}{2};
    B{i}{2} = B{i}{2} + B_delta{i}{2};
end

for i = 1:L
    for j = 1:size(W{i}{2},1)
        for k = 1:size(W{i}{2},2)
            if W{i}{2}(j,k) < 0, W{i}{2}(j,k) = W{i}{2}(j,k) - W_delta{i}{2}(j,k);end
            %             if W{i}{2}(j,k) < 0, W{i}{2}(j,k) = 0;end
        end
    end
    
    for j = 1:size(B{i}{2},1)
        for k = 1:size(B{i}{2},2)
            if B{i}{2}(j,k) < 0, B{i}{2}(j,k) = B{i}{2}(j,k) - B_delta{i}{2}(j,k);end
            %             if B{i}{2}(j,k) < 0, B{i}{2}(j,k) = 0;end
        end
    end
end

for cd_cell = 1:type
    for j = 1:size(C{cd_cell}{2},1)
        for k = 1:size(C{cd_cell}{2},2)
            if C{cd_cell}{2}(j,k) < 0, C{cd_cell}{2}(j,k) = C{cd_cell}{2}(j,k) - C_delta{cd_cell}{2}(j,k);end
            %         if C{2}(j,k) < 0, C{2}(j,k) =0;end
        end
    end
    if D{cd_cell}(1,2) < 0, D{cd_cell}(1,2) = D{cd_cell}(1,2) - D_delta{cd_cell}(1,2);end
end
% if D(1,2) < 0, D(1,2) = 0;end
end

%% Output W
function outputW(type,W,B,C,D,L)
for j = 1:L
    fprintf('At level %d, Mu of W is %.3e. \n',j,norm(W{j}{1}));
    %     mu = W{j}{1}
    fprintf('At level %d, Sigma of W is %.3e. \n',j,norm(W{j}{2}));
    %    sigma=W{j}{2}
    
    fprintf('At level %d, Mu of B is %.3e. \n',j,norm(B{j}{1}));
    %    mu = B{j}{1}
    fprintf('At level %d, Sigma of B is %.3e. \n',j,norm(B{j}{2}));
    %    sigma=B{j}{2}
end

for i = 1:type
    fprintf('Mu of C{%d} is %.3e. \n',i,norm(C{i}{1}));
    %mu = C{1}
    fprintf('Sigma of C{%d} is %.3e. \n',i,norm(C{i}{2}));
    %sigma=C{2}
    
    fprintf('Mu of D{i} is %.3e. \n',norm(D{i}(1)));
    %mu = D(1,1)
    fprintf('Sigma of D{i} is %.3e. \n',norm(D{i}(2)));
    %sigma=D(1,2)
end
end

function f = funmuoutput(type,net,funAct,funSoft,mu,L)
x = cell(1,L+1);
x{1} = 1;
d = 1;
for i = 1:L
    x{i+1} = zeros(net(1,i),1);
    if i == 1, xl = d;
    else xl = net(1,i-1);
    end
    for m = 1:net(1,i)
        for l = 1:xl
            x{i+1}(m) = x{i+1}(m) + mu .* x{i}(l);
        end
    end
    x{i+1} = funAct(x{i+1}+mu);
end
f = zeros(1);
for m = 1:net(1,L)
    f = f + 1 .* x{L+1}(m);
end
f = funSoft(f + 1);
f = f*type;
end