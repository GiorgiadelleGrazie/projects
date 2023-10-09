clear all
close all
clc 

%Circle

%Exercise 1 

%set the parameters k and sigma.
k = 10; 

%k = 20;
%k = 40;
sigma = 1;

%loading and plot the dataset.
data = load('Circle.mat');                    
X = data.X;                         
X_t = X';
% check the number of points. 
N = size(X,1); 

scatter(X(:,1),X(:,2)); 
title('Dataset Circle');                      
xlabel('x')
ylabel('y')

% the square norm of the points in order to avoid the
%computation of the square root, to speed up the process.
norm_2 = @(X) (X(1,:).*X(1,:)+X(2,:).*X(2,:));
similarity = @(X1, X2,sigma) exp(-norm_2(X2 - X1)./ 2*sigma^2);

% spalloc command  create an all-zero sparse matrix W
%of size N-by-N with room to hold N*k*2 nonzero elements.
W = spalloc(N,N,N*k*2);
%compute the adjacency matrix:
for i=1:N
    s_ij = similarity(X_t(:,i),X_t,sigma);% computethe similarity between the current point
                                          % (column i in the matrix X_t) and all the other points in the dataset 
    [values,idx] = maxk(s_ij,k+1); %Select the k largest values from the similarity matrix s_ij:
                                   % values will hold the maximum values, 
                                   % and idx will hold the corresponding indices of these values in the s_ij matrix.
    
  
   % For the current point i, populate the B matrix by inserting 
   % similarity function values into positions specified by the idx vector, excluding the value 
   % i itself (which corresponds to the current node undergoing similarity calculation).
    W(i,idx(idx ~= i)) = values(idx ~= i);
    W(idx(idx ~= i),i) = values(idx ~= i); % To maintain matrix symmetry,  mirror the same values within column i
end

%the plot of the k-nearest neighborhood similarity graph.
figure
gplot(W, X)

%the plot of the sparsity pattern of W.
figure
spy(W)



%Exercise 2

%the degree matrix D.
d = sum(W,2);
D = diag(d);
%the Laplacian.
L = D-W;



%Exercise 3
%  compute the 10 smallest  value eigenvalues
% and corresponding eigenvectors of the Laplacian matrix 
% of a graph, sort them, and then count 
% how many connected components there are in the graph based on the number of eigenvalues
% that are <1.0e-6.

[eigV,eigD] = eigs(L,10,"smallestabs");
eigD=diag(eigD);
%The eigenvectors are reordered based on the order of the eigenvalues using
[eigD,ij] = sort(eigD);
% the corresponding eigenvectors.
eigV = eigV(:,ij);
connected_comp = sum(eigD<1.0e-6);
s = sprintf("There are %d connected components",connected_comp);
disp(s)

%Exercise 4

% Calculate a set of lower eigenvalues of matrix 
% L and leverage their magnitudes for determining 
% an appropriate number of clusters 'M' for the points data sets.
eigD
figure
plot(eigD)


% So a suitable number of cluster:
M = 3;
%M = 2;         %for k=20.
%M = 2;         %for k=40.


%Exercise 5

% M eigenvectors corresponding to the smallest M eigenvalues are computed. 
% Given that these are the least values, 
% the first M columns of the previously derived eigenvectors matrix are considered.
U = eigV(:,1:M);



%Exercise 6


%We cluster the points with the k-means.
labels = kmeans(U,M);



%Exercise 7 

A = cell(1, M); % Initialize a cell array to store cluster points

for i = 1:M
    cluster_points = X_t(:, labels == i); % Select points belonging to cluster 'i'
    A{i} = cluster_points;                % Store the cluster points in cell array 'A'
end



%Exercise 8

% set markers to assign colors to clusters.
markers(1,:) = 'c*';
markers(2,:) = 'md';
markers(3,:) = 'g*';
markers(4,:) = 'bx';
markers(5,:) = 'cs';
markers(6,:) = 'ro';

% plot the results.
figure 
hold on 
for i=1:M
    plot(X_t(1,labels == i),X_t(2,labels == i), markers(i,:))
end
hold off 



%Exercise 9

%Compute the clusters with other clustering
%methods for the same set of points


%K-means:
idx_km = kmeans(X,M);

figure
hold on 
for i=1:M
    plot(X_t(1,idx_km == i),X_t(2,idx_km == i), markers(i,:))
end

hold off



%DBSCAN:
figure
hold on
idx_db = dbscan(X,0.75,5);            
            
gscatter(X_t(1,:),X_t(2,:),idx_db);

legend('off');
hold off
%%

%SPIRAL

%%

%Spiral
%Exercise 1 

%set the parameters k and sigma.
k = 10; 

%k = 20;
%k = 40;
sigma = 1;

%loading and plot the dataset.
data = load('Spiral.mat');  
%select only the first two columns 
%that contains the coordinates x and y of the points.
X = data.X(:,1:2);
X_t = X';
% check the number of points. 
N = size(X,1); 

scatter(X(:,1),X(:,2)); 
title('Dataset Spiral');                      
xlabel('x')
ylabel('y')

% the square norm of the points in order to avoid the
%computation of the square root, to speed up the process.
norm_2 = @(X) (X(1,:).*X(1,:)+X(2,:).*X(2,:));
similarity = @(X1, X2,sigma) exp(-norm_2(X2 - X1)./ 2*sigma^2);

%With the spalloc command we create an all-zero sparse matrix W
%of size N-by-N with room to hold N*k*2 nonzero elements.
W = spalloc(N,N,N*k*2);

   %compute the adjacency matrix:
for i=1:N
    s_ij = similarity(X_t(:,i),X_t,sigma);  %  computethe similarity between the current point 
                                            % (column i in the matrix X_t) and all the other points in the dataset 
    [values,idx] = maxk(s_ij,k+1); %select the k largest values from the similarity matrix s_ij:
                                   % values will hold the maximum values, 
                                   % and idx will hold the corresponding indices of these values in the s_ij matrix.
    
  
   % For the current point i, populate the B matrix by inserting 
   % similarity function values into positions specified by the idx vector, excluding the value 
   % i itself (which corresponds to the current node undergoing similarity calculation).
    W(i,idx(idx ~= i)) = values(idx ~= i);
    W(idx(idx ~= i),i) = values(idx ~= i);   % To maintain matrix symmetry, we mirror the same values within column i
end

%the plot of the k-nearest neighborhood similarity graph.
figure
gplot(W, X)

%the plot of the sparsity pattern of W.
figure
spy(W)



%Exercise 2

%the degree matrix D.
d = sum(W,2);
D = diag(d);
%the Laplacian.
L = D-W;



%Exercise 3
%  compute the 10 smallest  value eigenvalues
% and corresponding eigenvectors of the Laplacian matrix 
% of a graph, sort them, and then count 
% how many connected components there are in the graph based on the number of eigenvalues
% that are <1.0e-6.

[eigV,eigD] = eigs(L,10,"smallestabs");
eigD=diag(eigD);
%The eigenvectors are reordered based on the order of the eigenvalues using
[eigD,ij] = sort(eigD);
% the corresponding eigenvectors.
eigV = eigV(:,ij);
connected_comp = sum(eigD<1.0e-6);
s = sprintf("There are %d connected components",connected_comp);
disp(s)

%Exercise 4


% Calculate a set of lower eigenvalues of matrix 
% L and leverage their magnitudes for determining 
% an appropriate number of clusters 'M' for the points data sets.
eigD
figure
plot(eigD)

% Suitable number of cluster for different k  :
M = 3;          %for k=10
%M = 3;         %for k=20.
%M = 6;         %for k=40.


%Exercise 5

% M eigenvectors corresponding to the smallest M eigenvalues are computed. 
% Given that these are the least values, 
% the first M columns of the previously derived eigenvectors matrix are considered.
U = eigV(:,1:M);



%Exercise 6


% cluster the points with the k-means.
labels = kmeans(U,M);



%Exercise 7 

A = cell(1, M); % Initialize a cell array to store cluster points

for i = 1:M
    cluster_points = X_t(:, labels == i); % Select points belonging to cluster 'i'
    A{i} = cluster_points; % Store the cluster points in cell array 'A'
end



%Exercise 8

%We set markers to assign colors to clusters.
markers(1,:) = 'md'; 
markers(2,:) = 'bp'; 
markers(3,:) = 'go'; 
markers(4,:) = 'rx'; 
markers(5,:) = 'kh'; 
markers(6,:) = 'bo'; 


%We plot our results.
figure 
hold on 
for i=1:M
    plot(X_t(1,labels == i),X_t(2,labels == i), markers(i,:))
end
hold off 


%Exercise 9

%Compute the clusters  with other clustering
%methods for the same set of points


%K-means:
idx_km = kmeans(X,M);

figure
hold on 
for i=1:M
    plot(X_t(1,idx_km == i),X_t(2,idx_km == i), markers(i,:))
end

hold off



%DBSCAN:
figure
hold on
            
idx_db = dbscan(X,2,5);             
gscatter(X_t(1,:),X_t(2,:),idx_db);
legend('off');
hold off
%%

%3D model 



%%
%plot the 3D dataset.

 % plot the 3D dataset.
z1 = linspace(15,50,1000);
x1 = exp(-z1./10).*sin(5*z1);
y1 = exp(-z1./10).*cos(5*z1);

z2 = 0:pi/50:30;
x2 = sin(z2);
y2 = cos(z2);

x = [x2,x1];
y=[y2,y1];
z = [z2,z1];
X = [x;y;z]';
X_t = X';
 %check the number of points.
 N = size(X,1);
 figure
 scatter3(x,y,z,'o')
 view(-30,10)

 %set the parameters k and sigma
k = 10;
%k = 20;
%k = 40;
 sigma = 1;

% the square norm of the points in order to avoid the
%computation of the square root, to speed up the process.
 norm_2 = @(X) (X(1,:).*X(1,:)+X(2,:).*X(2,:)+X(3,:).*X(3,:));
 similarity = @(X1, X2,sigma) exp(-norm_2(X2 - X1)./ 2*sigma^2);
 
 %With the spalloc command create an all-zero sparse matrix W
 %of size N-by-N with room to hold N*k*2 nonzero elements.
 W = spalloc(N,N,N*k*2);


   %compute the adjacency matrix:

 for i=1:N
     s_ij = similarity(X_t(:,i),X_t,sigma);  % computethe similarity between the current point (column i in the matrix X_t) 
                                             % and all the other points in the dataset 
     [values,idx] = maxk(s_ij,k+1);%select the k largest values from the similarity matrix s_ij:
                                   % values will hold the maximum values, 
                                   % and idx will hold the corresponding indices of these values in the s_ij matrix.

   % For the current point i, populate the B matrix by inserting 
   % similarity function values into positions specified by the idx vector, excluding the value 
   % i itself (which corresponds to the current node undergoing similarity calculation).
     W(i,idx(idx ~= i)) = values(idx ~= i);
     W(idx(idx ~= i),i) = values(idx ~= i); % To maintain matrix symmetry, we mirror the same values within column i
 end

 %the plot of the k-nearest neighborhood similarity graph.
figure
gplot(W, X)
 %plot the sparsity pattern of W.
 figure
 spy(W)


%Exercise 2

%the degree matrix D.
d = sum(W,2);
D = diag(d);
%the Laplacian.
L = (D^(-.5)) * (D-W) * (D^(-.5));

%Exercise 3
%  compute the 10 smallest  value eigenvalues
% and corresponding eigenvectors of the Laplacian matrix 
% of a graph, sort them, and then count 
% how many connected components there are in the graph based on the number of eigenvalues
% that are <1.0e-6.

[eigV,eigD] = eigs(L,10,"smallestabs");
eigD=diag(eigD);
[eigD,ij] = sort(eigD); %The eigenvectors are reordered based on the order of the eigenvalues using
eigV = eigV(:,ij); % the corresponding eigenvectors.
connected_comp = sum(eigD<1.0e-6);
s = sprintf("There are %d connected components",connected_comp);
disp(s)

%Exercise 4


% Calculate a set of lower eigenvalues of matrix 
% L and leverage their magnitudes for determining 
% an appropriate number of clusters 'M' for the points data sets.
eigD
figure
plot(eigD)



% So a suitable number of cluster should be for different k:
M = 2; %k=10
%M=3;  %k=20
%M=4;  %k=40

%Exercise 5

% M eigenvectors corresponding to the smallest M eigenvalues are computed. 
% Given that these are the least values, 
% the first M columns of the previously derived eigenvectors matrix are considered.
U = eigV(:,1:M);


%Exercise 6


% cluster the points with the k-means.
labels = kmeans(U,M);


%Exercise 7 

A = cell(1, M); % Initialize a cell array to store cluster points

for i = 1:M
    cluster_points = X_t(:, labels == i); % Select points belonging to cluster 'i'
    A{i} = cluster_points; % Store the cluster points in cell array 'A'
end

%Exercise 8

%We set markers to assign colors to clusters.
markers(1,:) = 'c*';
markers(2,:) = 'md';
markers(3,:) = 'g*';
markers(4,:) = 'bx';
markers(5,:) = 'cs';
markers(6,:) = 'ro';

%We plot our results.
 figure
 hold on 
 for i=1:M
     plot3(X_t(1,labels == i),X_t(2,labels == i),X_t(3,labels == i),'o')
 end 
 hold off
 view(3);
