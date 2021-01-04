clc; clear;
%%
% Name: Devosmita Chatterjee
% Assignment4

%training set
x = csvread('training_set.csv',0,0,[0 0 9999 1]);

target = csvread('training_set.csv',0,2,[0 2 9999 2]);
%%
%validation set
x1 = csvread('validation_set.csv',0,0,[0 0 4999 1]);

target1 = csvread('validation_set.csv',0,2,[0 2 4999 2]);

%%

M1 = 10;
M2 = 9;

m = 10000;
%%
V1 = zeros(M1,1);
V2 = zeros(M2,1);
%O = 0;
%%
error1 = zeros(M1,1);
error2 = zeros(M2,1);
%error3 = 0;

%%
theta1 = zeros(M1,1);
w1_update = zeros(M1,2);
theta1_update = zeros(M1,1);

theta2 = zeros(M2,1);
w2_update = zeros(M2,M1);
theta2_update = zeros(M2,1);

theta3 = 0;%
w3_update = zeros(M2,1);
theta3_update = 0;


%%

a=-4;
b=4;
w1 = (b-a).*rand(M1,2)+a;
w2 = (b-a).*rand(M2,M1)+a;
w3 = (b-a).*rand(1,M2)+a;

eta = 0.02;

%%
V11 = zeros(M1,m);
V21 = zeros(M2,m);
O1 = zeros(1,m);
out1 = zeros(1,m);

T = 10^6;
for t=1:T
    %%
    mu = randi([1 m]);
    
    for j=1:M1
        V1(j) = tanh(-theta1(j)+w1(j,:)*x(mu,:)');
    end
    for i=1:M2
        V2(i) = tanh(-theta2(i)+w2(i,:)*V1(:));
    end
    O =    tanh(-theta3+w3*V2(:));
    
    
    %%
    
    %calculate error for output layer
    error3 = (target(mu)-O).*(1-O.^2);
    
    for i=1:M2
        error2(i) = error3*w3(i).*(1-V2(i).^2);
    end
    
    for j=1:M1
        error1(j) = error2(:)'*w2(:,j)*(1-V1(j).^2);
    end
    
    %end
    
    %%
    
    for i = 1:M2
        %calculate weight update and add to weight
        w3_update(i) = eta*(error3*V2(i)');
        w3(i) = w3(i) + w3_update(i);
    end
    
    % calculate threshold update and subtract from threshold
    theta3_update = eta*sum(error3);
    theta3 = theta3 - theta3_update;
    
    %%
    for i = 1:M2
        
        for j = 1:M1
            %calculate weight update and add to weight
            w2_update(i,j) = eta*(error2(i)*V1(j)');
            w2(i,j) = w2(i,j) + w2_update(i,j);
        end
        
        % calculate threshold update and subtract from threshold
        theta2_update(i) = eta*sum(error2(i));
        theta2(i) = theta2(i) - theta2_update(i);
        
    end
    %%
    for j = 1:M1
        
        for k = 1:2
            %calculate weight update and add to weight
            w1_update(j,k) = eta*(error1(j)*x(mu,k));
            w1(j,k) = w1(j,k) + w1_update(j,k);
        end
        
        % calculate threshold update and subtract from threshold
        theta1_update(j) = eta*sum(error1(j));
        theta1(j) = theta1(j) - theta1_update(j);
        
    end
end

for mu = 1:5000
    for j=1:M1
        V11(j,mu) = tanh(-theta1(j)+w1(j,:)*x1(mu,:)');
    end
    for i=1:M2
        V21(i,mu) = tanh(-theta2(i)+w2(i,:)*V11(:,mu));
    end
    O1(mu) = tanh(-theta3+w3*V21(:,mu));
    
    if (O1(mu) < 0)
        out1(mu) = -1;
        %end
        %if (O1(mu) > 0||O(mu) == 0)
    else
        out1(mu) = +1;
    end
end

%% Experiment with different values of M1 and M2 and train the perceptron using the training set so that C is below 12%.

% classification error for the validation set
p_val = 5000;
C = 0;
for mu1 = 1:p_val
    C = abs(out1(mu1)-target1(mu1)) + C;
end

C = (1/(2*p_val))*C

csvwrite('w1.csv',w1);
csvwrite('w2.csv',w2);
csvwrite('w3.csv',w3');
csvwrite('t1.csv',theta1);
csvwrite('t2.csv',theta2);
csvwrite('t3.csv',theta3);
