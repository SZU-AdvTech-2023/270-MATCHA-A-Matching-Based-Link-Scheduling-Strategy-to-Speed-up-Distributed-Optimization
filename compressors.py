import torch

def get_top_k(x, ratio):
    """
    it will sample the top 1-ratio of the samples.
    获得数据中前 1- ratio 百分比 大 的数据
    """
    x_data = x.view(-1)
    x_len = x_data.nelement()
    top_k = max(1, int(x_len * (1 - ratio)))

    # get indices and the corresponding values
    if top_k == 1:
        _, selected_indices = torch.max(x_data.abs(), dim=0, keepdim=True)
    else:
        _, selected_indices = torch.topk(
            x_data.abs(), top_k, largest=True, sorted=False
        )
    return x_data[selected_indices], selected_indices