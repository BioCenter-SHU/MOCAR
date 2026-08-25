"""Dataset adapters and collation utilities for supported benchmarks."""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np

class IEMOCAPDataset(Dataset):
    def __init__(self, path='data/iemocap_multimodal_features.pkl', train=True):

        # videoText[vid]   -> [Li, 1024]
        # videoVisual[vid] -> [Li, 342]
        # videoAudio[vid]  -> [Li, 1582]
        # videoLabels[vid] -> [Li]

        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class IEMOCAP4Dataset(Dataset):
    """GraphSmile's IEMOCAP4 split, adapted to the 0701 trainer batch API.

    The source pickle contains exactly the official four emotions; no label
    filtering or remapping is applied here.  Feature dimensions are text=1024,
    visual=512 and audio=100.
    """
    def __init__(self, path='data/iemocap_multi_features_4.pkl', train=True):
        (self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,
         self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,
         self.testVid) = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = list(self.trainVid if train else self.testVid)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (torch.FloatTensor(np.asarray(self.videoText[vid])),
                torch.FloatTensor(np.asarray(self.videoVisual[vid])),
                torch.FloatTensor(np.asarray(self.videoAudio[vid])),
                torch.FloatTensor([[1, 0] if speaker == 'M' else [0, 1]
                                   for speaker in self.videoSpeakers[vid]]),
                torch.FloatTensor([1] * len(self.videoLabels[vid])),
                torch.LongTensor(np.asarray(self.videoLabels[vid])), vid)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else
                pad_sequence(dat[i], batch_first=True) if i < 6 else dat[i].tolist()
                for i in dat]

class MELDDataset(Dataset):
    def __init__(self, path, train=True):

        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):

        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class CMUMOSEISevenDataset(Dataset):
    """CMU-MOSEI regression labels discretized into seven sentiment classes."""

    _normalization_cache = {}

    def __init__(self, path='data/cmumosei_multi_regression_features.pkl', train=True):
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = list(self.trainVid if train else self.testVid)
        self.len = len(self.keys)
        if path not in self._normalization_cache:
            self._normalization_cache[path] = {
                'text': self._feature_stats(self.videoText, self.trainVid),
                'audio': self._feature_stats(self.videoAudio, self.trainVid),
                'visual': self._feature_stats(self.videoVisual, self.trainVid),
            }
        self.normalization_stats = self._normalization_cache[path]

    @staticmethod
    def _feature_stats(features, train_keys):
        total = None
        total_square = None
        count = 0
        for key in train_keys:
            values = np.asarray(features[key], dtype=np.float64)
            feature_sum = values.sum(axis=0)
            feature_square_sum = np.square(values).sum(axis=0)
            total = feature_sum if total is None else total + feature_sum
            total_square = (
                feature_square_sum
                if total_square is None
                else total_square + feature_square_sum
            )
            count += values.shape[0]
        mean = total / count
        variance = np.maximum(total_square / count - np.square(mean), 1e-12)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

    def _normalize(self, values, modality):
        mean, std = self.normalization_stats[modality]
        return (np.asarray(values, dtype=np.float32) - mean) / std

    @staticmethod
    def _to_seven_class(value):
        if value < -2:
            return 0
        if value < -1:
            return 1
        if value < 0:
            return 2
        if value == 0:
            return 3
        if value <= 1:
            return 4
        if value <= 2:
            return 5
        return 6

    def __getitem__(self, index):
        vid = self.keys[index]
        labels = [
            self._to_seven_class(value)
            for value in np.asarray(self.videoLabels[vid])
        ]
        return (
            torch.FloatTensor(self._normalize(self.videoText[vid], 'text')),
            torch.FloatTensor(self._normalize(self.videoVisual[vid], 'visual')),
            torch.FloatTensor(self._normalize(self.videoAudio[vid], 'audio')),
            torch.FloatTensor([
                [1, 0] if speaker == 'M' else [0, 1]
                for speaker in np.asarray(self.videoSpeakers[vid])
            ]),
            torch.FloatTensor([1] * len(labels)),
            torch.LongTensor(labels),
            vid,
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [
            pad_sequence(dat[i]) if i < 4
            else pad_sequence(dat[i], True) if i < 6
            else dat[i].tolist()
            for i in dat
        ]

class _PickleSplitCache:
    """Keep each large SIMS pickle loaded once per Python process."""

    _cache = {}

    @classmethod
    def load(cls, path):
        if path not in cls._cache:
            with open(path, 'rb') as handle:
                cls._cache[path] = pickle.load(handle)
        return cls._cache[path]

class SIMSRegressionDataset(Dataset):
    """CH-SIMS data using the original train/valid/test split."""

    def __init__(self, path='data/sims_unaligned_39.pkl', split='train'):
        if split not in {'train', 'valid', 'test'}:
            raise ValueError('split must be train, valid, or test')
        self.split = split
        self.data = _PickleSplitCache.load(path)[split]
        self.length = len(self.data['regression_labels'])
        self.target_length = 39

    @staticmethod
    def _truncate_from_first_valid(values, target_length):
        values = np.asarray(values, dtype=np.float32)
        nonzero = np.flatnonzero(np.any(values != 0, axis=1))
        start = int(nonzero[0]) if nonzero.size else 0
        result = values[start:start + target_length]
        if result.shape[0] < target_length:
            padding = np.zeros(
                (target_length - result.shape[0], values.shape[1]),
                dtype=np.float32,
            )
            result = np.concatenate([result, padding], axis=0)
        return result

    def __getitem__(self, index):
        text = np.asarray(self.data['text'][index], dtype=np.float32)
        audio = self._truncate_from_first_valid(
            self.data['audio'][index], self.target_length
        )
        vision = self._truncate_from_first_valid(
            self.data['vision'][index], self.target_length
        )
        valid = (
            np.any(text != 0, axis=1)
            | np.any(audio != 0, axis=1)
            | np.any(vision != 0, axis=1)
        ).astype(np.float32)
        if valid.sum() == 0:
            valid[0] = 1.0

        qmask = np.zeros((self.target_length, 2), dtype=np.float32)
        qmask[:, 0] = 1.0
        labels = {
            key: torch.tensor(
                float(self.data[
                    'regression_labels' if key == 'M'
                    else 'regression_labels_' + key
                ][index]),
                dtype=torch.float32,
            )
            for key in 'MTAV'
        }
        return {
            'text': torch.from_numpy(text),
            'vision': torch.from_numpy(vision),
            'audio': torch.from_numpy(audio),
            'qmask': torch.from_numpy(qmask),
            'umask': torch.from_numpy(valid),
            'labels': labels,
            'id': self.data['id'][index],
            'index': index,
        }

    def __len__(self):
        return self.length

class SIMS2RegressionDataset(Dataset):
    """CH-SIMS v2.0 data with the source mean-pooling protocol."""

    def __init__(self, path='data/sims2_unaligned_001.pkl', split='train'):
        if split not in {'train', 'valid', 'test'}:
            raise ValueError('split must be train, valid, or test')
        self.split = split
        self.data = _PickleSplitCache.load(path)[split]
        self.length = len(self.data['regression_labels'])

    @staticmethod
    def _mean_valid(values, length=None):
        values = np.asarray(values, dtype=np.float32)
        if length is not None:
            valid_length = max(1, min(int(length), values.shape[0]))
            valid_values = values[:valid_length]
        else:
            mask = np.any(values != 0, axis=1)
            valid_values = values[mask]
            if valid_values.shape[0] == 0:
                valid_values = values[:1]
        return valid_values.mean(axis=0, keepdims=True).astype(np.float32)

    def __getitem__(self, index):
        text = self._mean_valid(self.data['text'][index])
        audio = self._mean_valid(
            self.data['audio'][index],
            self.data['audio_lengths'][index],
        )
        vision = self._mean_valid(
            self.data['vision'][index],
            self.data['vision_lengths'][index],
        )
        labels = {
            key: torch.tensor(
                float(self.data[
                    'regression_labels' if key == 'M'
                    else 'regression_labels_' + key
                ][index]),
                dtype=torch.float32,
            )
            for key in 'MTAV'
        }
        return {
            'text': torch.from_numpy(text),
            'vision': torch.from_numpy(vision),
            'audio': torch.from_numpy(audio),
            'qmask': torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            'umask': torch.ones(1, dtype=torch.float32),
            'labels': labels,
            'id': self.data['id'][index],
            'index': index,
        }

    def __len__(self):
        return self.length

def sims_five_class_label(value):
    """Map the original [-1, 1] sentiment score to the standard five bins."""
    value = float(value)
    if value <= -0.7:
        return 0
    if value <= -0.1:
        return 1
    if value <= 0.1:
        return 2
    if value <= 0.7:
        return 3
    return 4

class SIMSFiveClassDataset(SIMSRegressionDataset):
    """CH-SIMS with fixed five-class labels."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['label'] = torch.tensor(
            sims_five_class_label(self.data['regression_labels'][index]),
            dtype=torch.long,
        )
        return sample

class SIMSThreeClassDataset(SIMSRegressionDataset):
    """CH-SIMS three-way sentiment labels supplied by the MMSA pickle.

    Label IDs follow the common convention used by DecAlign:
    0=negative, 1=neutral, 2=positive.
    """

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['label'] = torch.tensor(
            int(np.asarray(self.data['classification_labels'][index]).reshape(-1)[0]),
            dtype=torch.long,
        )
        return sample

class SIMS2FiveClassDataset(SIMS2RegressionDataset):
    """CH-SIMS v2.0 with fixed five-class labels."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['label'] = torch.tensor(
            sims_five_class_label(self.data['regression_labels'][index]),
            dtype=torch.long,
        )
        return sample

def sims_binary_label(value):
    """Map the original sentiment score using the standard Acc2 boundary."""
    return 0 if float(value) <= 0.0 else 1

class SIMSBinaryDataset(SIMSRegressionDataset):
    """CH-SIMS with fixed binary labels."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['label'] = torch.tensor(
            sims_binary_label(self.data['regression_labels'][index]),
            dtype=torch.long,
        )
        return sample

class SIMS2BinaryDataset(SIMS2RegressionDataset):
    """CH-SIMS v2.0 with fixed binary labels."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['label'] = torch.tensor(
            sims_binary_label(self.data['regression_labels'][index]),
            dtype=torch.long,
        )
        return sample
