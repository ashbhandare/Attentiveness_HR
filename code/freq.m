clc;
T=1; %1 second resolution
hr = csvread('C:\Users\Ash\Desktop\mobile sensing\project\hrcsv.csv');

N = length(hr);
fs = 1/T;
t = linspace(0,T,fs);
hop=N/4
overlap=N-hop
S = stft(hr,overlap);

%Max frequency to visualize
maxFreq = N/8;

%time vector
time = linspace(0,T,size(S,2));

%frequency vector
freq1= linspace(0,fs*maxFreq/N,size(S(1:maxFreq,:),1));

%set colour scale range (dB)
clims = [-100 60];

%plot the STFT heatmap
fig = figure;
imagesc(time,freq1,20*log10(abs(S(1:maxFreq,:))),clims)
colorbar
axis xy
xlabel('TIME (s.)')
ylabel('FREQUENCY (Hz.)')
title(['C4 GUITAR: MAGNITUDE SPECTROGRAM ANALYSIS']);

%Mfft = fft(M);
%f = fs/2*linspace(0,1,N/2+1)
%plot(f,2*abs(Mfft(1:N/2+1)));
