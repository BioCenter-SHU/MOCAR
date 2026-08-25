"""Training loop for binary CH-SIMS2.0 classification."""

import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from confusion_regularizer import ModalityWiseOnlineConfusion, compute_mocar_loss
from model import (
    MaskedKLDivLoss,
    MaskedNLLLoss,
    PGM,
    Transformer_Based_Sequence_Classification_Model,
)
from sims_classification_utils import (
    classification_metrics,
    clone_state_dict,
    format_metrics,
    prepare_batch,
    save_summary,
    set_seed,
)

def _base_loss(outputs, labels, output_mask, nll_loss, kl_loss):
    task_losses = PGM(nll_loss, kl_loss)._compute_task_losses(
        outputs, labels.view(-1), output_mask
    )
    return sum(task_losses) / len(task_losses)

def _run_epoch(model, loader, device, nll_loss, kl_loss, epoch, args,
               n_classes, optimizer=None, confusion_bank=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all, predictions_all = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            values = prepare_batch(batch, device)
            text, vision, audio, qmask, input_mask, labels, output_mask, lengths = values
            if training:
                optimizer.zero_grad()
            outputs = model(
                text, vision, audio, input_mask, qmask, lengths
            )
            loss = _base_loss(
                outputs, labels, output_mask, nll_loss, kl_loss
            )
            if training:
                mocar_loss, _ = compute_mocar_loss(
                    outputs,
                    labels,
                    output_mask,
                    confusion_bank,
                    epoch,
                    args,
                    n_classes,
                )
                loss = loss + mocar_loss
                loss.backward()
                optimizer.step()

            predictions = outputs[4].argmax(dim=-1).view(-1)
            labels_all.extend(labels.view(-1).cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            total_loss += loss.item()
    return (
        total_loss / max(len(loader), 1),
        classification_metrics(labels_all, predictions_all),
    )

def run_experiment(args, dataset_class, dataset_name, dimensions, n_classes=2):
    if args.hidden_dim % args.n_head != 0:
        raise ValueError('--hidden-dim must be divisible by --n-head')
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    if use_cuda:
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError('--gpu must identify an available CUDA device')
        torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu) if use_cuda else 'cpu')
    os.makedirs(os.path.dirname(args.result_json) or '.', exist_ok=True)
    results = []

    for seed in args.seeds:
        set_seed(seed)
        datasets = {
            split: dataset_class(args.data_path, split)
            for split in ('train', 'valid', 'test')
        }
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=(split == 'train'),
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda'),
            )
            for split, dataset in datasets.items()
        }
        model = Transformer_Based_Sequence_Classification_Model(
            dataset=dataset_name,
            D_text=dimensions[0],
            D_visual=dimensions[1],
            D_audio=dimensions[2],
            n_head=args.n_head,
            n_classes=n_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            temp=args.temp,
        ).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        nll_loss = MaskedNLLLoss()
        kl_loss = MaskedKLDivLoss()
        confusion_bank = ModalityWiseOnlineConfusion(
            n_classes,
            beta=args.conf_beta,
            device=device,
        )

        if args.eval_only:
            if args.checkpoint is None:
                raise ValueError('--eval-only requires --checkpoint')
            checkpoint = torch.load(args.checkpoint, map_location=device)
            state_dict = (
                checkpoint['model_state_dict']
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict)
            test_loss, test_metrics = _run_epoch(
                model, loaders['test'], device, nll_loss, kl_loss,
                0, args, n_classes,
            )
            print(
                'checkpoint={} test_loss={:.4f} TEST [{}]'.format(
                    args.checkpoint, test_loss, format_metrics(test_metrics)
                ),
                flush=True,
            )
            results.append({
                'seed': seed,
                'checkpoint': args.checkpoint,
                'test_loss': test_loss,
                **test_metrics,
            })
            continue

        best_valid = -1.0
        best_epoch = 0
        best_state = None
        for epoch in range(args.epochs):
            train_loss, _ = _run_epoch(
                model, loaders['train'], device, nll_loss, kl_loss,
                epoch, args, n_classes, optimizer, confusion_bank,
            )
            valid_loss, valid_metrics = _run_epoch(
                model, loaders['valid'], device, nll_loss, kl_loss,
                epoch, args, n_classes,
            )
            print(
                'seed={} epoch={} train_loss={:.4f} valid_loss={:.4f} '
                'valid=[{}]'.format(
                    seed,
                    epoch + 1,
                    train_loss,
                    valid_loss,
                    format_metrics(valid_metrics),
                ),
                flush=True,
            )
            if valid_metrics['weighted_f1'] > best_valid:
                best_valid = valid_metrics['weighted_f1']
                best_epoch = epoch + 1
                best_state = clone_state_dict(model)
            if epoch + 1 - best_epoch >= args.early_stop_patience:
                break

        model.load_state_dict(best_state)
        test_loss, test_metrics = _run_epoch(
            model, loaders['test'], device, nll_loss, kl_loss,
            best_epoch - 1, args, n_classes,
        )
        print(
            'seed={} best_epoch={} test_loss={:.4f} TEST [{}]'.format(
                seed, best_epoch, test_loss, format_metrics(test_metrics)
            ),
            flush=True,
        )
        result = {
            'seed': seed,
            'best_epoch': best_epoch,
            'best_valid_weighted_f1': best_valid,
            'test_loss': test_loss,
            **test_metrics,
        }
        results.append(result)
        if not args.no_save_checkpoint:
            os.makedirs(args.save_best_dir, exist_ok=True)
            run_name = args.run_name or '{}_seed{}'.format(dataset_name.lower(), seed)
            checkpoint_path = os.path.join(args.save_best_dir, '{}_best.pt'.format(run_name))
            torch.save({
                'model_state_dict': best_state,
                'args': vars(args),
                'dataset': dataset_name,
                'n_classes': n_classes,
                'seed': seed,
                'best_epoch': best_epoch,
                'best_valid_weighted_f1': best_valid,
                'test_metrics': test_metrics,
            }, checkpoint_path)
            print('saved_checkpoint={}'.format(checkpoint_path), flush=True)

    save_summary(
        args.result_json,
        {'dataset': dataset_name, 'runs': results},
    )
