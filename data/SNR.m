
% y = data.measured_pin_location';
%         signal = y;

close all;
clear all;
data = jsondecode(fileread('C:\Users\Owner\Downloads\data_6_with_FFT.json'));

% Coble Data     
y_test = xlsread("y_test.csv")';
y_test = y_test(:,1);
y_predict_2ms = xlsread("y_predict - 2.5ms.csv")';
y_predict_3ms = xlsread("y_predict - 3ms.csv")';
y_predict_5ms = xlsread("y_predict - 5ms.csv")';
y_predict_10ms = xlsread("y_predict - 10ms.csv")';

%Raw Data
x = data.acceleration_data';
y = data.measured_pin_location';
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
ypred_2ms = subsample_snr(signal, time, time_sub, y_predict_2ms)
ypred_3ms = subsample_snr(signal, time, time_sub, y_predict_3ms)
ypred_5ms = subsample_snr(signal, time, time_sub, y_predict_5ms)
ypred_10ms = subsample_snr(signal, time, time_sub, y_predict_10ms)





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