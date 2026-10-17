from __future__ import annotations

import torch


def multi_kernel_mmd(source, target, kernel_multiplier=2.0, kernel_count=5, fixed_bandwidth=None):
    combined = torch.cat((source, target), dim=0)
    squared_distance = ((combined[:, None, :] - combined[None, :, :]) ** 2).sum(dim=2)
    sample_count = combined.size(0)
    if fixed_bandwidth is None:
        denominator = max(sample_count * sample_count - sample_count, 1)
        bandwidth = squared_distance.detach().sum() / denominator
    else:
        bandwidth = torch.as_tensor(fixed_bandwidth, device=combined.device, dtype=combined.dtype)
    bandwidth = bandwidth.clamp_min(1e-8) / kernel_multiplier ** (kernel_count // 2)
    kernels = sum(
        torch.exp(-squared_distance / (bandwidth * kernel_multiplier**index))
        for index in range(kernel_count)
    )
    source_count = source.size(0)
    source_source = kernels[:source_count, :source_count]
    target_target = kernels[source_count:, source_count:]
    source_target = kernels[:source_count, source_count:]
    target_source = kernels[source_count:, :source_count]
    return source_source.mean() + target_target.mean() - source_target.mean() - target_source.mean()
