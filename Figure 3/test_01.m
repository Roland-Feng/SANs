%normal approximating exponential

global norm_type
norm_type = 2;
sampleNo = 7;
N = 10000;
if sampleNo == 1
    label{1} = exprnd(2,1,N);
    mean_label = mean(label{1});
elseif sampleNo == 2
    label1 = normrnd(1,1,1,N);
    label2 = normrnd(4,1,1,N);
    label{1} = label1;
    label{2} = label2;
    mean_label = mean([label1,label2]);   
elseif sampleNo ==3
    label{1} = normrnd(1,1,1,N);
    label{2} = normrnd(4,1,1,N);
    label{3} = normrnd(7,1,1,N);
    mean_label = mean(cell2mat(label));
elseif sampleNo == 4
    %%label{1} = normrnd(0.5,0.5,1,N).*normrnd(2,0.5,1,N) + normrnd(0.5,0.2,1,N) + normrnd(1,0.1,1,N);
    %label{1} = normrnd(0.5,0.5,1,N).*normrnd(2,1,1,N);
    label{1}=normrnd(1,0.5,1,N).*normrnd(2,1,1,N)+exprnd(2,1,N);
    mean_label = mean(label{1});
elseif sampleNo ==5
    label{1}=rand(1,N);
    mean_label = mean(label{1});
elseif sampleNo == 6
    labelTemp = zeros(1,N);
    temp = 0;
    while(temp~=N)
        sampleTemp = exprnd(2,1,1);
        if (sampleTemp<10)
            temp = temp+1;
            labelTemp(1,temp) = sampleTemp;
        end
    end
    label{1} = labelTemp;
    mean_label = mean(label{1});
elseif sampleNo == 7
    label{1} = normrnd(2,1,1,N);
    mean_label = mean(label{1});
elseif sampleNo == 8
    label{1} = exprnd(2,1,N)+normrnd(1,1,1,N);
    mean_label = mean(label{1});
end

%% Net
net = [3,3,3];
Maxiter = 2000;
%% error threshold
errorThreshold = 0.03;
%% delta
delta.start = 0.1;
delta.rate = 0.9;
delta.Step = 100;
%% learing rate
lr.start = 0.05;
lr.rate = 0.9;
lr.Step = 100;

%% Learning
[flag,loss,Model,f_Output] = randNN_01(N,label,Maxiter,net,errorThreshold,delta,lr,1,[]);