def count_parameters_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool):
    result = (out_channels * (in_channels * (kernel_size ** 2) + 1))
    return result if bias else result - out_channels
