"""Confusion-aware regularization used by MOCAR training tasks."""

import torch
import torch.nn.functional as F

MODALITY_NAMES = ('text', 'audio', 'video', 'fused')

def _masked_flatten(log_prob, labels, umask):

    mask = umask.view(-1).bool()
    flat_log_prob = log_prob.view(-1, log_prob.size(-1))
    flat_labels = labels.view(-1)
    return flat_log_prob[mask], flat_labels[mask]

def observed_true_rows(labels, umask, n_classes):
    mask = umask.view(-1).bool()
    flat_labels = labels.view(-1)[mask]
    rows = torch.zeros(n_classes, dtype=torch.bool, device=labels.device)
    if flat_labels.numel() > 0:
        rows[flat_labels.unique()] = True
    return rows

def soft_confusion_from_log_prob(log_prob, labels, umask, n_classes, temp=1.0):

    flat_log_prob, flat_labels = _masked_flatten(log_prob, labels, umask)
    if flat_labels.numel() == 0:
        return log_prob.new_zeros(n_classes, n_classes)

    probs = F.softmax(flat_log_prob / max(temp, 1e-12), dim=-1)
    one_hot = F.one_hot(flat_labels, n_classes).to(probs.dtype)

    c_num = one_hot.transpose(0, 1) @ probs
    c_den = one_hot.sum(dim=0).view(-1, 1).clamp(min=1.0)
    c = c_num / c_den
    c = c.clone()
    c.fill_diagonal_(0.0)
    return c

def soft_confusion_statistics(log_prob, labels, umask, n_classes, temp=1.0):
    flat_log_prob, flat_labels = _masked_flatten(log_prob, labels, umask)
    if flat_labels.numel() == 0:
        return (
            log_prob.new_zeros(n_classes, n_classes),
            log_prob.new_zeros(n_classes),
        )

    probs = F.softmax(flat_log_prob / max(temp, 1e-12), dim=-1)
    one_hot = F.one_hot(flat_labels, n_classes).to(probs.dtype)
    confusion_sum = one_hot.transpose(0, 1) @ probs
    class_count = one_hot.sum(dim=0)

    confusion_sum = confusion_sum.clone()
    confusion_sum.fill_diagonal_(0.0)
    return confusion_sum, class_count

def spectral_norm_loss(confusion_matrix, class_weight=None):

    if class_weight is not None:
        w = class_weight.to(confusion_matrix.device, dtype=confusion_matrix.dtype)
        confusion_matrix = confusion_matrix * w.view(1, -1)
    return torch.linalg.matrix_norm(confusion_matrix, ord=2)

def pair_margin_loss(log_prob, labels, umask, hard_pairs, confusion_matrix,
                     base_margin=0.2, margin_scale=0.5):

    if not hard_pairs:
        return log_prob.new_tensor(0.0)

    flat_log_prob, flat_labels = _masked_flatten(log_prob, labels, umask)
    if flat_labels.numel() == 0:
        return log_prob.new_tensor(0.0)

    losses = []
    for true_cls, conf_cls, _ in hard_pairs:
        sample_mask = flat_labels == true_cls
        if not sample_mask.any():
            continue

        margin = base_margin + margin_scale * confusion_matrix[true_cls, conf_cls].detach()
        gold_score = flat_log_prob[sample_mask, true_cls]
        conf_score = flat_log_prob[sample_mask, conf_cls]
        losses.append(F.relu(margin - gold_score + conf_score).mean())

    if not losses:
        return log_prob.new_tensor(0.0)
    return torch.stack(losses).mean()

class ModalityWiseOnlineConfusion:

    def __init__(self, n_classes, beta=0.9, device='cpu'):
        self.n_classes = n_classes
        self.beta = beta
        self.device = device
        self.matrices = {
            name: torch.zeros(n_classes, n_classes, device=device)
            for name in MODALITY_NAMES
        }

        self.initialized = {
            name: torch.zeros(n_classes, dtype=torch.bool, device=device)
            for name in MODALITY_NAMES
        }
        self.confusion_sums = {
            name: torch.zeros(n_classes, n_classes, device=device)
            for name in MODALITY_NAMES
        }
        self.class_counts = {
            name: torch.zeros(n_classes, device=device)
            for name in MODALITY_NAMES
        }

    def update(self, modality_log_probs, labels, umask, temp=1.0):
        # modality_log_probs: {'text': t_log_prob, 'audio': a_log_prob, ...}
        current = {}
        observed_rows = observed_true_rows(labels, umask, self.n_classes)
        for name, log_prob in modality_log_probs.items():
            c_batch = soft_confusion_from_log_prob(
                log_prob,
                labels,
                umask,
                self.n_classes,
                temp=temp,
            )
            current[name] = c_batch
            batch_sum, batch_count = soft_confusion_statistics(
                log_prob,
                labels,
                umask,
                self.n_classes,
                temp=temp,
            )

            if observed_rows.any():
                old_init = self.initialized[name]
                old_sum = self.confusion_sums[name].detach()
                old_count = self.class_counts[name].detach()
                new_sum = old_sum.clone()
                new_count = old_count.clone()

                first_seen = observed_rows & (~old_init)
                seen_before = observed_rows & old_init
                if first_seen.any():
                    new_sum[first_seen] = batch_sum.detach()[first_seen]
                    new_count[first_seen] = batch_count.detach()[first_seen]
                if seen_before.any():
                    new_sum[seen_before] = (
                        self.beta * old_sum[seen_before]
                        + batch_sum.detach()[seen_before]
                    )
                    new_count[seen_before] = (
                        self.beta * old_count[seen_before]
                        + batch_count.detach()[seen_before]
                    )

                new_matrix = self.matrices[name].detach().clone()
                new_matrix[observed_rows] = (
                    new_sum[observed_rows]
                    / new_count[observed_rows].unsqueeze(1).clamp(min=1e-12)
                )
                new_matrix.fill_diagonal_(0.0)

                self.confusion_sums[name] = new_sum
                self.class_counts[name] = new_count
                self.matrices[name] = new_matrix
                self.initialized[name] = old_init | observed_rows

        return current

    def hard_pairs(self, name, topk=5, min_score=0.0):

        c = self.matrices[name]
        pairs = []
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i == j:
                    continue
                score = float(c[i, j].detach().cpu())
                if score > min_score:
                    pairs.append((i, j, score))
        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs[:topk]

    def top_pairs_summary(self, topk=3):
        return {
            name: self.hard_pairs(name, topk=topk)
            for name in MODALITY_NAMES
        }

def compute_mocar_loss(outputs, labels, umask, confusion_bank, epoch, args, n_classes):

    modality_log_probs = {
        'text': outputs[0],
        'audio': outputs[1],
        'video': outputs[2],
        'fused': outputs[3],
    }

    current_confusions = confusion_bank.update(
        modality_log_probs,
        labels,
        umask,
        temp=args.conf_temp,
    )

    car_loss = outputs[3].new_tensor(0.0)
    margin_loss = outputs[3].new_tensor(0.0)
    if epoch + 1 > args.conf_warmup:
        modality_weights = {
            'text': args.car_text_weight,
            'audio': args.car_audio_weight,
            'video': args.car_video_weight,
            'fused': args.car_fused_weight,
        }
        for name in MODALITY_NAMES:
            car_loss = car_loss + modality_weights[name] * spectral_norm_loss(current_confusions[name])
            hard_pairs = confusion_bank.hard_pairs(
                name,
                topk=args.topk_conf_pairs,
                min_score=args.conf_min_score,
            )
            margin_loss = margin_loss + pair_margin_loss(
                modality_log_probs[name],
                labels,
                umask,
                hard_pairs,
                confusion_bank.matrices[name],
                base_margin=args.base_margin,
                margin_scale=args.margin_scale,
            )
        margin_loss = margin_loss / len(MODALITY_NAMES)

    mocar_loss = args.car_lambda * car_loss + args.pair_margin_lambda * margin_loss
    log_info = {
        'mocar_loss': float(mocar_loss.detach().cpu()),
        'car_loss': float(car_loss.detach().cpu()),
        'pair_margin_loss': float(margin_loss.detach().cpu()),
    }
    return mocar_loss, log_info
