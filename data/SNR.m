
% y = data.measured_pin_location';
%         signal = y;

close all;
clear all;
data = jsondecode(fileread('data_6_with_FFT.json'));
% Coble Data     
y_test = csvread("y_test.csv")';
y_test = y_test(:,1);
y_predict_2_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\2.5ms.csv")';
y_predict_3ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\3ms.csv")';
y_predict_3_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\3.5ms.csv")';
y_predict_4ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\4ms.csv")';
y_predict_4_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\4.5ms.csv")';
y_predict_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\5ms.csv")';
y_predict_5_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\5.5ms.csv")';
y_predict_6ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\6ms.csv")';
y_predict_6_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\6.5ms.csv")';
y_predict_7ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\7ms.csv")';
y_predict_7_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\7.5ms.csv")';
y_predict_8ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\8ms.csv")';
y_predict_8_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\8.5ms.csv")';
y_predict_9ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\9ms.csv")';
y_predict_9_5ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\9.5ms.csv")';
y_predict_10ms = csvread("\Users\dncob\Documents\GitHub\Real-time-LSTM\data\input_rate_outputs\10ms.csv")';

%Raw Data
x = data.acceleration_data';
y = csvread("C:\Users\dncob\Documents\GitHub\Real-time-LSTM\data\y_test.csv")';
y = y(1,:);
signal = y;
time = data.time_acceleration_data';

% delete nan values from pin position
nanidx=find(isnan(y));
for i=nanidx
    legitidx=find(~isnan(y(1:i)));
    last_legitidx=legitidx(end);
    y(i)=y(last_legitidx);
end

%Resampling for zero order hold
[L,M] = rat(numel(x)/numel(y));
time = resample(time,M,L);

[L,M] = rat(numel(y_test)/numel(time));
time_sub = resample(time,L,M);

%Find SNR(dB)
SNR_2_5ms = subsample_snr(signal, time, time_sub, y_predict_2_5ms)
SNR_3ms = subsample_snr(signal, time, time_sub, y_predict_3ms)
SNR_3_5ms = subsample_snr(signal, time, time_sub, y_predict_3_5ms)
SNR_4ms = subsample_snr(signal, time, time_sub, y_predict_4ms)
SNR_4_5ms = subsample_snr(signal, time, time_sub, y_predict_4_5ms)
SNR_5ms = subsample_snr(signal, time, time_sub, y_predict_5ms)
SNR_5_5ms = subsample_snr(signal, time, time_sub, y_predict_5_5ms)
SNR_6ms = subsample_snr(signal, time, time_sub, y_predict_6ms)
SNR_6_5ms = subsample_snr(signal, time, time_sub, y_predict_6_5ms)
SNR_7ms = subsample_snr(signal, time, time_sub, y_predict_7ms)
SNR_7_5ms = subsample_snr(signal, time, time_sub, y_predict_7_5ms)
SNR_8ms = subsample_snr(signal, time, time_sub, y_predict_8ms)
SNR_8_5ms = subsample_snr(signal, time, time_sub, y_predict_8_5ms)
SNR_9ms = subsample_snr(signal, time, time_sub, y_predict_9ms)
SNR_9_5ms = subsample_snr(signal, time, time_sub, y_predict_9_5ms)
SNR_10ms = subsample_snr(signal, time, time_sub, y_predict_10ms)





function SNR = subsample_snr(signal, time, time_sub, y_pred)
    
    subsample_snr = [];
    signal_pred_zoh = myzoh(time,time_sub,double(y_pred));
    % compute error signal from zero order hold
    error_signal = signal - signal_pred_zoh;
    % snip lead-in for MLP
        
    error_signal_snip = error_signal;
    signal_snip = signal;
        
    
    % compute error power
%         error_power = rms(error_signal_snip)^2;
        % compute signal power
%         signal_power = rms(signal_snip)^2;
    rmean = [];
    emean = [];
        for i=1:(numel(signal_snip))
            if isnan(signal_snip(i)) == 1
                rmean = [rmean, (signal_snip(i+1))^2];
            else
                rmean = [rmean, (signal_snip(i))^2];
            end
            
            
            if isnan(error_signal_snip(i)) == 1
                emean = [emean, (error_signal_snip(i+1))^2];
            else
                emean = [emean, (error_signal_snip(i))^2];
            end
            
        end    
        signal_power = sqrt(sum(rmean)/numel(signal_snip));
        error_power = sqrt(sum(emean)/numel(error_signal_snip));


        % compute SNR
        SNR = [subsample_snr,log10(signal_power / error_power) * 20];
        SNR = abs(SNR);
end 
        
        

function signal = myzoh (x,x_sub,signal_in)
    
    signal = zeros(1,size(x,2));
    
    i_sub = 1;
        
    for i=1:size(x,2)
        signal(1,i) = signal_in(1,i_sub);
        
        while (x(1,i) >= x_sub(1,i_sub)) && (i_sub<size(signal_in,2))
            i_sub = i_sub+1;
        end
        
        
    end

end