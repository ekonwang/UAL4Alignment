import torch
import torch.nn.functional as F
import numpy as np

def label_smooth(labels, classes, smoothing=0.1):
    """
    Applies label smoothing to the given labels.
    
    Args:
        labels (Tensor): A tensor containing the labels.
        classes (int): Total number of classes.
        smoothing (float): Smoothing factor.
        
    Returns:
        Tensor: A new tensor with smoothed labels.
    """
    original_device = labels.device
    if isinstance(smoothing, list):
        assert len(smoothing) == labels.size(0)

        labels_copy = labels.clone()
        labels_copy[labels_copy == -1] = 0

        smoothed_list = []
        for i, smooth_value in enumerate(smoothing):
            confidence = 1.0 - smooth_value
            smooth_label = torch.full(size=(labels.size(1), classes), fill_value=smooth_value / (classes - 1)).to(labels.device)
            smooth_label.scatter_(1, labels_copy[i].unsqueeze(-1), confidence)
            smoothed_list.append(smooth_label)
        smooth_label = torch.stack(smoothed_list)

    else:
        # Create a tensor with smoothing/num_classes for each label
        confidence = 1.0 - smoothing

        # offload for saving some GPU memory
        labels = labels.cpu()
        smooth_label = torch.full(size=(labels.size(0), labels.size(1), classes), fill_value=smoothing / (classes - 1)).to(labels.device)
        # set labels = 0 where labels == -1, in case of CUDA insertion error
        labels_copy = labels.clone()
        labels_copy[labels_copy == -1] = 0
        smooth_label.scatter_(2, labels_copy.unsqueeze(-1), confidence)
    
    return smooth_label.to(original_device)


def loss_fn(logits, targets_, smoothing=0.0):
    # TODO: support mask inputs for label smoothing
    targets = label_smooth(targets_, logits.size(-1), smoothing=smoothing)
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:, :].contiguous()

    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1, targets.size(-1))

    real_mask = (targets_ != -1)
    mask = real_mask.unsqueeze(-1).expand(-1, -1, logits.size(-1))[..., 1:, :]
    mask = mask.view(-1, logits.size(-1)).float()

    log_preds = F.log_softmax(logits, dim=1)
    log_preds = log_preds * mask
    loss = -torch.sum(log_preds * targets) / real_mask.float().sum()
    # import pdb; pdb.set_trace()
    return loss


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param:.3f}%"
    )


def make_score_dist(raw_scores, target_mean=0.1, max_values=0.99):
    assert target_mean <= max_values

    scores = np.array(raw_scores)
    while(abs(np.mean(scores) - target_mean) >= 0.001):
        scores = scores / np.mean(scores) * target_mean
        scores = np.clip(scores, scores.min(), max_values)
        print(np.mean(scores))    
    
    return scores.tolist()

