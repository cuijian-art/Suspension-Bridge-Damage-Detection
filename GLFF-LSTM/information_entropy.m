% 计算信息熵
function entropy = information_entropy(X)
    % 计算X的概率分布
    probs = histcounts(X, 'Normalization', 'probability');

    % 计算信息熵
    entropy = -sum(probs .* log2(probs), 'omitnan');
end
