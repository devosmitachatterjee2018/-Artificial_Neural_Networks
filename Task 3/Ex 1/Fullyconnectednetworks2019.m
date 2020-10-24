clc; clear;
%%
% Name: Devosmita Chatterjee
% Assignment 3.1
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(1);
xTrain = xTrain-mean(xTrain,2);
xValid = xValid-mean(xTrain,2);
xTest = xTest-mean(xTrain,2);

[C_Train1,C_Valid1,C_Test1,lastepoch1] = network1(xTrain,tTrain,xValid,tValid,xTest,tTest);
[C_Train2,C_Valid2,C_Test2,lastepoch2] = network2(xTrain,tTrain,xValid,tValid,xTest,tTest);
[C_Train3,C_Valid3,C_Test3,lastepoch3] = network3(xTrain,tTrain,xValid,tValid,xTest,tTest);
[C_Train4,C_Valid4,C_Test4,lastepoch4] = network4(xTrain,tTrain,xValid,tValid,xTest,tTest);
%%
disp(lastepoch1)
disp(C_Train1(lastepoch1))
disp(C_Valid1(lastepoch1))
disp(C_Test1)
%9
%0.4254
%0.4912
%0.4910
%%
disp(lastepoch2)
disp(C_Train2(lastepoch2))
disp(C_Valid2(lastepoch2))
disp(C_Test2)
%9
%0.3936
%0.4588
%0.4577
%%
disp(lastepoch3)
disp(C_Train3(lastepoch3))
disp(C_Valid3(lastepoch3))
disp(C_Test3)
%15
%0.3139
%0.4715
%0.4711
%%
disp(lastepoch4)
disp(C_Train4(lastepoch4))
disp(C_Valid4(lastepoch4))
disp(C_Test4)
%18
%0.2978
%0.4481
%0.4520
%%
x = 1:20;
y1 = C_Train1;
y2 = C_Valid1;
y3 = C_Train2;
y4 = C_Valid2;
y5 = C_Train3;
y6 = C_Valid3;
y7 = C_Train4;
y8 = C_Valid4;

plot(x,y1,'b')
hold on
plot(x,y2,'r')
hold on
plot(x,y3,'m')
hold on
plot(x,y4,'g')
hold on
plot(x,y5,'y')
hold on
plot(x,y6,'c')
hold on
plot(x,y7,'k')
hold on
plot(x,y8,'color',[0.9100    0.4100    0.1700])

set(gca, 'YScale', 'log')
xlabel('Number of epochs')
ylabel('Classification errors')

legend('C\_Train1','C\_Valid1','C\_Train2','C\_Valid2','C\_Train3','C\_Valid3','C\_Train4','C\_Valid4','Location', 'Best')
