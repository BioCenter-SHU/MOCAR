"""Train or evaluate MOCAR on the four-class IEMOCAP task."""

import os
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(SRC_DIR))

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# IEMOCAP4 is the only active training path.  MELDDataset is retained solely
# because the copied upstream helper `get_MELD_loaders` remains below intact.
from dataloader import IEMOCAP4Dataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model, PGM

from confusion_regularizer import ModalityWiseOnlineConfusion, compute_mocar_loss
from runtime import configure_run

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
torch.backends.cudnn.enabled = False

def set_seed(seed=61078):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def _extend_params_if_exists(params, module, attr_name):

    if hasattr(module, attr_name):
        params.extend(getattr(module, attr_name).parameters())

def collect_pgm_shared_params(model):

    params = []
    for attr_name in [
        'shared_transformer',
        'textf_input', 'acouf_input', 'visuf_input',
        'text_to_audio', 'text_to_vision', 'bias_gate_audio', 'bias_gate_vision',
        't_t', 'a_a', 'v_v',
        't_t_gate', 't_gate', 'a_a_gate', 'a_gate', 'v_v_gate', 'v_gate',
        'features_reduce_t', 'features_reduce_a', 'features_reduce_v',
    ]:
        _extend_params_if_exists(params, model, attr_name)
    return params

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):

    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):

    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def get_IEMOCAP4_loaders(data_path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    # Isolated GraphSmile-compatible IEMOCAP4 data source and split.
    trainset = IEMOCAP4Dataset(path=data_path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAP4Dataset(path=data_path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False,
                        confusion_bank=None, n_classes=None):

    losses, preds, labels, masks = [], [], [], []
    PGM_loss = PGM(loss_function, kl_loss)

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-1]]

        qmask = qmask.permute(1, 0, 2)

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        outputs = model(textf, visuf, acouf, umask, qmask, lengths)

        labels_ = label.view(-1)

        shared_params = collect_pgm_shared_params(model)

        individual_losses = PGM_loss._compute_task_losses(outputs, labels_, umask)

        weights = torch.ones(len(individual_losses), device = individual_losses[0].device) / len(individual_losses)

        loss = sum(w * task_loss for w, task_loss in zip(weights, individual_losses))

        mocar_log_info = None
        if train and args.use_mocar:
            mocar_loss, mocar_log_info = compute_mocar_loss(
                outputs,
                label,
                umask,
                confusion_bank,
                epoch,
                args,
                n_classes,
            )
            loss = loss + mocar_loss

        all_prob = outputs[4]
        pred_ = torch.argmax(all_prob.view(-1, all_prob.size()[2]), 1)

        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())

        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())

        if train:
            loss.backward()
            if args.tensorboard:

                for param in model.named_parameters():
                    if param[1].grad is not None:
                        writer.add_histogram(param[0], param[1].grad, epoch)

                writer.add_scalar('weights/task1_weight', weights[0], epoch)
                writer.add_scalar('weights/task2_weight', weights[1], epoch)
                writer.add_scalar('weights/task3_weight', weights[2], epoch)

                for i, task_loss in enumerate(losses):
                    writer.add_scalar(f'losses/task{i + 1}_loss', task_loss.item(), epoch)
                writer.add_scalar('losses/total_loss', loss.item(), epoch)

                if mocar_log_info is not None:
                    writer.add_scalar('mocar/mocar_loss', mocar_log_info['mocar_loss'], epoch)
                    writer.add_scalar('mocar/car_loss', mocar_log_info['car_loss'], epoch)
                    writer.add_scalar('mocar/pair_margin_loss', mocar_log_info['pair_margin_loss'], epoch)

            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore

def evaluate_loaded_model(model, dataloader):

    preds, labels, masks = [], [], []
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-1]]
            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

            outputs = model(textf, visuf, acouf, umask, qmask, lengths)
            all_prob = outputs[4]
            pred_ = torch.argmax(all_prob.view(-1, all_prob.size()[2]), 1)

            preds.append(pred_.data.cpu().numpy())
            labels.append(label.view(-1).data.cpu().numpy())
            masks.append(umask.view(-1).cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_accuracy, labels, preds, masks, avg_fscore

def format_mocar_pairs(confusion_bank, topk=3):
    if confusion_bank is None:
        return ''
    summary = confusion_bank.top_pairs_summary(topk=topk)
    parts = []
    for modality, pairs in summary.items():
        if pairs:
            pair_text = ', '.join(['{}->{}:{:.4f}'.format(i, j, score) for i, j, score in pairs])
        else:
            pair_text = 'none'
        parts.append('{} [{}]'.format(modality, pair_text))
    return ' | '.join(parts)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use when CUDA is available')
    parser.add_argument('--Dataset', default='IEMOCAP4', choices=['IEMOCAP4'], help='isolated four-class IEMOCAP task')
    parser.add_argument('--data-path', default='data/iemocap_multi_features_4.pkl', help='pre-extracted feature pickle')

    parser.add_argument('--lr', type=float, default=2.0566274895060898e-4, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=1.7468952853162445e-6, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.60, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=200, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=61078, help='random seed')

    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help='stop if monitored W-F1 does not improve for N consecutive epochs; <=0 disables early stopping')
    parser.add_argument('--hidden_dim', type=int, default=512, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--temp', type=int, default=2, metavar='temp', help='temp')

    parser.add_argument('--rank', type=int, default=16, metavar='rank', help='projection rank')
    parser.add_argument('--order', type=int, default=3, metavar='order', help='fusion order')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--save-best-dir', default='outputs/iemocap4/checkpoints', help='directory for best checkpoint')
    parser.add_argument('--log-dir', default='outputs/iemocap4/logs', help='directory for console logs')
    parser.add_argument('--no-save-checkpoint', action='store_true', default=False,
                        help='do not write model checkpoint files')
    parser.add_argument('--run-name', default=None, help='checkpoint name prefix')
    parser.add_argument('--checkpoint', default=None, help='checkpoint path to load')
    parser.add_argument('--eval-only', action='store_true', default=False, help='only evaluate a loaded checkpoint')

    parser.add_argument('--use-mocar', action='store_true', default=True,
                        help='enable modality-wise online confusion regularization')

    parser.add_argument('--conf-beta', type=float, default=0.85,
                        help='EMA beta for online modality-wise confusion matrices')

    parser.add_argument('--conf-temp', type=float, default=1.0,
                        help='temperature for building soft confusion matrices')

    parser.add_argument('--conf-warmup', type=int, default=5,
                        help='epochs for updating confusion matrices without adding MOCAR loss')

    parser.add_argument('--car-lambda', type=float, default=0.04,
                        help='weight for modality-wise CAR spectral regularizer')

    parser.add_argument('--pair-margin-lambda', type=float, default=0.03,
                        help='weight for hard-pair margin loss')

    parser.add_argument('--topk-conf-pairs', type=int, default=5,
                        help='top-k directed hard emotion pairs per modality')

    parser.add_argument('--conf-min-score', type=float, default=0.0,
                        help='minimum EMA confusion score for a hard pair')

    parser.add_argument('--base-margin', type=float, default=0.0,
                        help='base margin for hard-pair boundary loss')

    parser.add_argument('--margin-scale', type=float, default=0.0,
                        help='extra margin scale multiplied by online confusion score')

    parser.add_argument('--car-text-weight', type=float, default=0.5,
                        help='CAR spectral weight for text confusion matrix')

    parser.add_argument('--car-audio-weight', type=float, default=1.0,
                        help='CAR spectral weight for audio confusion matrix')

    parser.add_argument('--car-video-weight', type=float, default=0.5,
                        help='CAR spectral weight for video confusion matrix')

    parser.add_argument('--car-fused-weight', type=float, default=1.5,
                        help='CAR spectral weight for fused confusion matrix')

    parser.add_argument('--conf-log-interval', type=int, default=0,
                        help='print top modality-wise hard pairs every N epochs when MOCAR is enabled')

    args = parser.parse_args()
    set_seed(args.seed)
    if args.run_name is None:
        args.run_name = '{}_seed{}'.format(args.Dataset.lower(), args.seed)
    configure_run(args, 'iemocap4')
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError('--gpu must be in [0, {}], got {}'.format(torch.cuda.device_count() - 1, args.gpu))
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
        print('Running on GPU {}'.format(device))
    else:
        device = torch.device('cpu')
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    # GraphSmile IEMOCAP4 feature contract: text=1024, visual=512, audio=100.
    D_audio = 100
    D_visual = 512
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 2
    n_classes = 4

    print('temp {}'.format(args.temp))

    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,args.rank,args.order, n_classes=n_classes, hidden_dim=args.hidden_dim, n_speakers=n_speakers, dropout=args.dropout)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    model.to(device)

    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    confusion_bank = None
    if args.use_mocar:
        confusion_bank = ModalityWiseOnlineConfusion(
            n_classes=n_classes,
            beta=args.conf_beta,
            device=device,
        )
        print('MOCAR enabled: modality-wise online confusion matrices + CAR + hard-pair margin')

    # Do not reuse six-class class weights: IEMOCAP4 has a distinct label
    # distribution.  The GraphSmile reference uses unweighted NLL.
    loss_function = MaskedNLLLoss()
    train_loader, valid_loader, test_loader = get_IEMOCAP4_loaders(args.data_path,
        valid=0.0, batch_size=batch_size, num_workers=0)

    if args.checkpoint is not None:

        map_location = device
        checkpoint = torch.load(args.checkpoint, map_location=map_location)
        model_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(model_state_dict)
    if args.eval_only:

        if args.checkpoint is None:
            raise ValueError('--eval-only requires --checkpoint')

        test_acc, test_label, test_pred, test_mask, test_fscore = evaluate_loaded_model(model, test_loader)
        print('Checkpoint test performance..')
        print('Acc: {}'.format(test_acc))
        print('F-Score: {}'.format(test_fscore))
        print(classification_report(test_label, test_pred, sample_weight=test_mask, digits=4))
        print(confusion_matrix(test_label, test_pred, sample_weight=test_mask))

        if args.tensorboard:
            writer.close()
        exit()

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    best_acc = None

    best_monitor_fscore = None
    epochs_without_improvement = 0
    early_stop_patience = args.early_stop_patience
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(
            model,
            loss_function,
            kl_loss,
            train_loader,
            e,
            optimizer,
            True,
            confusion_bank=confusion_bank,
            n_classes=n_classes,
        )
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, kl_loss, test_loader, e)

        all_fscore.append(test_fscore)
        monitor_fscore = valid_fscore
        monitor_split = 'valid'
        if np.isnan(monitor_fscore):
            monitor_fscore = test_fscore
            monitor_split = 'test'

        is_best = False

        if best_fscore == None or best_fscore <= test_fscore:
            if best_fscore == None:
                best_fscore = test_fscore
                best_acc = test_acc
                best_label, best_pred, best_mask = test_label, test_pred, test_mask
                is_best = True

            elif best_acc < test_acc and best_fscore == test_fscore:
                best_acc = test_acc
                best_fscore = test_fscore
                best_label, best_pred, best_mask = test_label, test_pred, test_mask
                is_best = True

            elif best_fscore < test_fscore:
                best_acc = test_acc
                best_fscore = test_fscore
                best_label, best_pred, best_mask = test_label, test_pred, test_mask
                is_best = True

        if is_best and not args.no_save_checkpoint:

            os.makedirs(args.save_best_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.save_best_dir, '{}_best.pt'.format(args.run_name))
            torch.save({
                'epoch': e + 1,
                'best_acc': best_acc,
                'best_fscore': best_fscore,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args).copy(),
                'dataset': args.Dataset,
                'seed': args.seed,
                'classification_report': classification_report(
                    best_label,
                    best_pred,
                    sample_weight=best_mask,
                    digits=4
                ),
                'confusion_matrix': confusion_matrix(
                    best_label,
                    best_pred,
                    sample_weight=best_mask
                ).tolist(),

                'mocar_confusion_matrices': {
                    name: matrix.detach().cpu().tolist()
                    for name, matrix in confusion_bank.matrices.items()
                } if confusion_bank is not None else None,

            }, checkpoint_path)

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   test_fscore, round(time.time() - start_time, 2)))
        if early_stop_patience > 0:
            if best_monitor_fscore is None or monitor_fscore > best_monitor_fscore:
                best_monitor_fscore = monitor_fscore
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stop_patience:
                break

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
