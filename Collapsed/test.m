clear;
d = 10;
k = 10;
n = 200;
% load sonar
% k = max(y);

r = 1.5;
[X,y] = kmeansRnd(d,k,n);
m = softseeds(X,round(k*r));

[~,init] = max(bsxfun(@minus,m'*X,sum(m.^2)'/2));

% [~,~,bound1]=vbigm(X,init);
% [~,~,bound2]=acvbfdigm(X,init);
[~,~,bound3]=svebfdigm(X,init);

% t = [length(bound1),length(bound2)];
t = [length(bound1),length(bound2),length(bound3)];
% bound = zeros(3,max(t));
% bound(1,1:t(1)) = bound1;
% bound(2,1:t(2)) = bound2;
% bound(3,1:t(3)) = bound3;

step = 30;
strade = round(max(t)/step);
idx1 = 1: strade : t(1);
idx2 = 1: strade : t(2);
idx3 = 1: strade : t(3);
%%
plot(idx1, bound1(idx1), '-gd', 'LineWidth', 2, 'MarkerFaceColor', 'y', 'MarkerSize',8);
hold on
plot(idx2, bound2(idx2), '-bs', 'LineWidth', 2,  'MarkerFaceColor', 'y', 'MarkerSize',8);
plot(idx3, bound3(idx3), '-ro', 'LineWidth', 2,  'MarkerFaceColor', 'y', 'MarkerSize',8);
xlabel('number of iterations');ylabel('lower bound');
% legend('VBGM','CVBGM');

legend('VBGM','CVBGM','SVBGM');
grid on
hold off