"""Train or evaluate MOCAR on the binary CH-SIMS2.0 task."""

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from dataloader import SIMS2BinaryDataset
from sims_binary_training import run_experiment
from runtime import configure_run

def parse_args():
    parser = argparse.ArgumentParser(description='Binary CH-SIMS2.0 training.')
    parser.add_argument(
        '--data-path',
        default='data/sims2_unaligned_001.pkl',
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.30)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stop-patience', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.8170277510731414e-4)
    parser.add_argument('--l2', type=float, default=1.1242474349963723e-4)
    parser.add_argument('--temp', type=float, default=4.0)
    parser.add_argument('--conf-beta', type=float, default=0.9)
    parser.add_argument('--conf-temp', type=float, default=1.0)
    parser.add_argument('--conf-warmup', type=int, default=0)
    parser.add_argument('--car-lambda', type=float, default=0.002)
    parser.add_argument('--pair-margin-lambda', type=float, default=0.001)
    parser.add_argument('--topk-conf-pairs', type=int, default=1)
    parser.add_argument('--conf-min-score', type=float, default=0.0)
    parser.add_argument('--base-margin', type=float, default=0.05)
    parser.add_argument('--margin-scale', type=float, default=0.01)
    parser.add_argument('--car-text-weight', type=float, default=1.0)
    parser.add_argument('--car-audio-weight', type=float, default=1.0)
    parser.add_argument('--car-video-weight', type=float, default=1.0)
    parser.add_argument('--car-fused-weight', type=float, default=2.0)
    parser.add_argument('--result-json', default='outputs/sims2/results.json')
    parser.add_argument('--save-best-dir', default='outputs/sims2/checkpoints')
    parser.add_argument('--log-dir', default='outputs/sims2/logs')
    parser.add_argument('--run-name', default='mocar_sims2')
    parser.add_argument('--checkpoint', default=None, help='checkpoint path to load')
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help='only evaluate a loaded checkpoint')
    parser.add_argument('--no-save-checkpoint', action='store_true', default=False,
                        help='do not write the best-validation checkpoint')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    configure_run(args, 'sims2')
    run_experiment(
        args,
        SIMS2BinaryDataset,
        'SIMS2_BINARY',
        (768, 177, 25),
    )
