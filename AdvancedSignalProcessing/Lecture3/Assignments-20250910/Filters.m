
clear all
clc

% Read audio

% noise:
[d,fs1] = audioread('Noise.wav') ;
% Signal
[x,fs1] = audioread('Music.wav') ;
% Noisy signal
[u,Fs] = audioread('Noisy_Music.wav') ;

N = length(u) ;
% Filter Order:
M = 12 ;

% *************************************************************************
% *************************************************************************
% *************************************************************************



% LMS
tic
mu = 0.01 ;
w = randn(M,1) ;
padded_u = [zeros(M-1,1) ; u] ;
y = zeros(N,1) ;
for n=1:round(N)
    u_vect = padded_u(n:n+M-1) ;
    e = d(n) - w'*u_vect ;
    w = w + mu*e*u_vect ;
    y(n) = w'*u_vect ;
end

% y2 = [zeros(sampleDelay,1) ; y] ;
filtered_signal_LMS = u - y ;
toc


% *************************************************************************

% NLMS
tic
mu = 1 ;
w = randn(M,1) ;
padded_u = [zeros(M-1,1) ; u] ;
y = zeros(N,1) ;
Eps = 0.0001 ;
for n=1:round(N)
    u_vect = padded_u(n:n+M-1) ;
    mu1 = mu/(Eps + norm(u_vect)^2) ;
    %norm(u_vect)^2
    e = d(n) - w'*u_vect ;
    w = w + mu1*e*u_vect ;
    y(n) = w'*u_vect ;
end

% y2 = [zeros(sampleDelay,1) ; y] ;
filtered_signal_NLMS = u - y ;
toc


% *************************************************************************

% RLS:
tic
lambda = 1 - 1/(0.1*M) ;
delta = 0.01 ;
P = 1/delta*eye(M) ;
w = randn(M,1) ;
padded_u = [sqrt(delta)*randn(M-1,1) ; u] ;
y = zeros(N,1) ;

for n=1:N
    u_vect = padded_u(n:n+M-1) ;
    PI = P*u_vect ;
    gain_k = PI/(lambda + u_vect'*PI) ;
    prior_error = d(n) - w'*u_vect ;
    w = w + prior_error*gain_k ;
    P = P/lambda - gain_k*(u_vect'*P)/lambda ;
    y(n) = w'*u_vect ;
end
% y1 = [zeros(sampleDelay,1) ; y] ;
filtered_signal_RLS = u - y ;
toc

