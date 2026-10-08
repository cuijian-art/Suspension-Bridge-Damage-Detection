function ee = EnergyEntropy(y, window_size)
% Calculate the energy entropy of a signal

% Input:   y: time series;
%          window_size: size of the window for energy calculation

% Output: 
%          ee: energy entropy

ly = length(y);
num_windows = floor(ly / window_size); % 计算窗口的个数

energy = zeros(1, num_windows); % 存储每个窗口的能量值

for i = 1:num_windows
    start_index = (i - 1) * window_size + 1;
    end_index = start_index + window_size - 1;
    window = y(start_index:end_index); % 提取窗口数据
    energy(i) = sum(window.^2); % 计算窗口的能量值
end

total_energy = sum(energy); % 计算信号的总能量

probabilities = energy / total_energy; % 计算每个窗口的能量概率

ee = -sum(probabilities .* log2(probabilities)); % 计算能量熵

end
