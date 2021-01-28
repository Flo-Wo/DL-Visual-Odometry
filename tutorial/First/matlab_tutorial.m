%% math operations
4+5
19-12
5*123
125/12
5^4
sqrt(125)

%% logical operations
11 >= 10 % 11 is greater and equal than 10
12 ~= 11 % 12 is not equal to 11
1 && 0 % 1 and 0
1 || 0 % 1 or 0

%% variables
a = 10; % semicolon to stop print out
disp(a);

b = 'hello world'; % string
disp(b);

A = [1 2; 3 4; 5 6]; % 3-by-2 matrix

V = [1 ; 2 ; 3];

A = 1:0.1:2; % matrix start from 1 to 2, increment with a step 0.1

A = 1:6; % matrix start from 1 to 6, incremet with a step 1 (default)

A = ones(4,5); % generate a matrix with size 4-by-5 and each element is 1

A = 2*ones(4,5); % generate a matrix with size 4-by-5 and each element is 2

A = zeros(4,5); % generate a matrix with size 4-by-5 and each element is 0

A = rand(5,6); % generate a random matrix with size 5-by-6

A = eye(4); % generate a 4-by-4 identity matrix

%% data save and load
A = [1 2; 3 4; 5 6];
save data A;
load('data.mat');
clear % clear variables in workspace
clc % clear command window

%% matrix access and manipulation
A = [1 2; 3 4; 5 6];

disp(length(A)); % give the size of longer dimension

disp(A(3,2)); % access the element 3rd row, 2nd column in A
disp(A(:,2)); % access all the elements of 2nd column in A
disp(A(1,:)); % access all the elements of 1st row in A
disp(A([1 3],:)); % access all the elements of 1st and 3rd row in A
A(:,2) = [10;11;12]; % replace 2 column of A with [10; 11; 12]
A(:,2) = []; % remove 2 column
A = [A, [100; 101; 102]]; % add one more column to right of A

B = [10 11; 12 13; 14 15];
C = [A B];
disp(C);

[rownum,colnum] = size(C); % acquire the row and column number of matrix
C = C(:); % put a matrix into a vector
disp(C);
disp(reshape(C,rownum,colnum)); % reshape it into a matrix


%% data computation

A = [1 2; 3 4; 5 6]; % 3-by-2 matrix
B = [11 12; 13 14; 15 16]; % 3-by-2 matrix
C = [1 1; 2 2]; % 2-by-2 matrix

disp(A + B) % matrix addition
disp(A - B) % matrix subtraction
disp(A*C) % matrix multipliation
disp(A.*B) % dot product
disp(A.^2) % power 2 each element in A
disp(1./A) % 1 divide each element
disp(exp(A)) % exponential each element
disp(abs(A)) % absolute value of each element
disp(A+5) % each element plus 5
disp(A') % tranpose of A or transpose(A)


a = [1.2 14.9 2.4 0.5];
disp(max(a));
disp(min(a));
disp(sum(a));
disp(prod(a));
disp(round(a));
disp(floor(a));
disp(ceil(a));
[r,c] = find(a == min(a)); % find the minimum number in a
[r,c] = find(a<3);

disp(rand(3,3));
A = magic(3);
disp(A)
max(A) % max(A,[],2) 
min(A)
sum(A)
sum(sum(eye(3).*A))


%% plot data
figure(1)
t = 0:0.01:0.98;
y1 = sin(2*pi*4*t);
y2 = cos(2*pi*4*t);
plot(t,y1); hold on;
plot(t,y2);
xlabel('time');
ylabel('value');
title('my plot');
legend('sin','cos');
%axis([0 2*pi -1.5 1.5]) % x axis from 0 to 2pi, and y axis from -1.5 to 1.5

figure(2)
subplot(2,1,1)
plot(t,y1);
xlabel('time');
ylabel('value');
title('sins output');

subplot(2,1,2)
plot(t,y2);
xlabel('time');
ylabel('value');
title('cosine output');

%% controll statement
% plus number from 1 to 100
% for loop
output = 0;
for i = 1:100
    output = output + i;
end
disp(output);

% while
output = 0;
i = 1;
while i <= 100
    output = output + i;
    i = i + 1;
end
disp(output);


% if...else
input = 5;
if input == 1
    disp('The input number is one');
elseif input == 2
    disp('The input number is two')
else
    disp('Unknown number');
end


%% write your function
x = 6;
[y1,y2] = squareAndCubThisNumber(x);
disp([num2str(y1) '; ' num2str(y2)]);


%% vectorization

a = 1:100000;
c = 0;
tic
for i = 1:size(a,2)
    c = c + a(i)*a(i);
end
toc

tic
d = a*a';
toc



