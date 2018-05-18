%% Summary the results


net{1} = {[2],[3],[4],[5],[6]};
%net{1}={[2],[3]};
net{2} = {[2,2],[3,3],[4,4],[5,5],[6,6]};
net{3} = {[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]};
N = 20;

for j=1:length(net)
    for k=1:length(net{j})
        fprintf('### The net is %d  %d\n', j,k);
        [flag{j,k},loss{j,k},Model{j,k},f_Output{j,k},rate{j,k}] = test(net{j}{k},N);
    end
end
save results;