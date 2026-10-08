clc
clear
close all
warning off
%% 导入数据
load('.\y.mat');% 读取.DAT中的数据

%% 提取特征
fs=12000;%采样频率
Ts=1/fs;%采样周期
L=1024;%采样点数
t=(0:L-1)*Ts;%时间序列
N=1024;%样本点数
[mm , nn]= size(y); % 数据尺寸大小

for j=1:mm  % 遍历各个样本提取特征
    % 时域特征
    [ timestruct(j) ] = timeDomainFeatures(y(j,:));
    % 频域特征
    [ frequencystruct(j) ] = frequencyDomainFeatures( y(j,:),fs);
    % 小波包特征
    [ waveletstruct(j) ] =  waveletFeatures( y(j,:));
    % 熵特征
    [ Entropystruct(j) ] = EntropyFeatures(y(j,:));
end

%% 合并所有特征

% 初始化合并后的结构体数组
allFeatures = struct();

for j = 1:mm
    % 合并时域特征
    fieldsTime = fieldnames(timestruct(j));
    for k = 1:length(fieldsTime)
        allFeatures(j).(fieldsTime{k}) = timestruct(j).(fieldsTime{k});
    end
    
    % 合并频域特征
    fieldsFreq = fieldnames(frequencystruct(j));
    for k = 1:length(fieldsFreq)
        allFeatures(j).(fieldsFreq{k}) = frequencystruct(j).(fieldsFreq{k});
    end
    
    % 合并小波包特征
    fieldsWavelet = fieldnames(waveletstruct(j));
    for k = 1:length(fieldsWavelet)
        allFeatures(j).(fieldsWavelet{k}) = waveletstruct(j).(fieldsWavelet{k});
    end
    
    % 合并熵特征
    fieldsEntropy = fieldnames(Entropystruct(j));
    for k = 1:length(fieldsEntropy)
        allFeatures(j).(fieldsEntropy{k}) = Entropystruct(j).(fieldsEntropy{k});
    end
end

%% 以部分特征为例绘制特征图象
figure(1);  % 时域
subplot(3,3,1);feature = [timestruct.kurtosis];plot(feature,'r');title('峭度');
subplot(3,3,2);feature = [timestruct.peak];plot(feature,'r');title('峰值');
subplot(3,3,3);feature = [timestruct.std];plot(feature,'r');title('标准差');
subplot(3,3,4);feature = [timestruct.pulseFactor];plot(feature,'r');title('脉冲因子');
subplot(3,3,5);feature = [timestruct.marginFactor];plot(feature,'r');title('裕度因子');
subplot(3,3,6);feature = [timestruct.skewness];plot(feature,'r');title('偏度');
subplot(3,3,7);feature = [timestruct.shapeFactor];plot(feature,'r');title('波形因子');
subplot(3,3,8);feature = [timestruct.peakingFactor];plot(feature,'r');title('峰值因子');
subplot(3,3,9);feature = [timestruct.clearanceFactor];plot(feature,'r');title('余隙因子');

figure(2);  % 频域
subplot(2,2,1);feature = [frequencystruct.MF];plot(feature,'b');title('平均频率');
subplot(2,2,2);feature = [frequencystruct.FC];plot(feature,'b');title('重心频率');
subplot(2,2,3);feature = [frequencystruct.RMSF];plot(feature,'b');title('频率均方根');
subplot(2,2,4);feature = [frequencystruct.RVF];plot(feature,'b');title('频率标准差');

figure(3);  % 熵
subplot(2,2,1);feature = [Entropystruct.sample];plot(feature,'r');title('样本熵');
subplot(2,2,2);feature = [Entropystruct.fuzzy];plot(feature,'r');title('模糊熵');
subplot(2,2,3);feature = [Entropystruct.appro];plot(feature,'r');title('近似熵');
subplot(2,2,4);feature = [Entropystruct.infor];plot(feature,'r');title('信息熵');

figure(4);  % 小波包特征
subplot(2,2,1);feature = [waveletstruct.p1];plot(feature,'k');title('小波子带能量比');
subplot(2,2,2);feature = [waveletstruct.energyE];plot(feature,'k');title('小波能量熵');
subplot(2,2,3);feature = [waveletstruct.E1];plot(feature,'k');title('小波尺度熵');
subplot(2,2,4);feature = [waveletstruct.qyshang];plot(feature,'k');title('小波奇异谱熵');
