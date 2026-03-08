from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from data_factory.logPreprocess_helper import Preprocessor
from utils.get_random_state import get_random_state
import logging

class HDFSSegLoader(object):

    def __init__(self, ensemble_param, data_path, win_size, step, data_seq_len, mode="train"):

        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Create preprocessor for loading data
        preprocessor = Preprocessor(
            length=data_seq_len,  # Extract sequences of h items
            timeout=float('inf'),  # Do not include a maximum allowed time between events
        )

        path_train = data_path + '/hdfs_train.txt'
        path_test_abnormal = data_path + '/hdfs_test_abnormal.txt'
        path_test_normal = data_path + '/hdfs_test_normal.txt'

        # Load data from HDFS dataset
        X_train, _, _, _ = preprocessor.text(path_train, verbose=True)
        X_test, _, _, _ = preprocessor.text(path_test_normal, verbose=True)
        X_test_anomaly, _, _, _ = preprocessor.text(path_test_abnormal, verbose=True)

        data = X_train.numpy()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        if self.mode == "train":
            # Bootstrap sample
            random_number = get_random_state('./model_config/bat_config/ensemble_train_hdfs_config.yaml', ensemble_param[0], ensemble_param[1], ensemble_param[2], ensemble_param[3])
            data, _ = resample(data, data, replace=True, n_samples=len(data), random_state=random_number)

        test_normal = X_test
        test_abnormal = X_test_anomaly
        test_data = np.concatenate((test_normal, test_abnormal), axis=0)
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        test_normal_labels = np.full([len(test_normal), 1], 0, dtype=int)
        test_abnormal_labels = np.full([len(test_abnormal), 1], 1, dtype=int)
        self.test_labels = np.concatenate((test_normal_labels, test_abnormal_labels), axis=None)

        logging.info(f"test data shape: {self.test.shape}")
        logging.info(f"train data shape: {self.train.shape}")
        logging.info(f"test_labels shape: {self.test_labels.shape}")


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class BGLSegLoader(object):

    def __init__(self, ensemble_param, data_path, win_size, step, data_seq_len, mode="train"):
        
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Create preprocessor for loading data
        preprocessor = Preprocessor(
            length=data_seq_len,      # Extract sequences of length h
            timeout=float("inf"),     # No maximum allowed time between events
        )

        # Dataset paths
        path_train = data_path + "/bgl_train.txt"
        path_test_abnormal = data_path + "/bgl_test_abnormal.txt"
        path_test_normal = data_path + "/bgl_test_normal.txt"

        # Load data from BGL dataset
        X_train, _, _, _ = preprocessor.text(path_train, verbose=True)
        X_test, _, _, _ = preprocessor.text(path_test_normal, verbose=True)
        X_test_anomaly, _, _, _ = preprocessor.text(path_test_abnormal, verbose=True)

        # Fit scaler on training data
        data = X_train.numpy()
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Bootstrap sampling for training mode
        # ensemble_param: num_epochs, k, e_layer_num, batch_size
        if self.mode == "train":
            random_number = get_random_state(
                "./model_config/bat_config/ensemble_train_bgl_config.yaml",
                ensemble_param[0],
                ensemble_param[1],
                ensemble_param[2],
                ensemble_param[3],
            )
            data, _ = resample(
                data,
                data,
                replace=True,
                n_samples=len(data),
                random_state=random_number,
            )

        # Prepare test data
        test_normal = X_test
        test_abnormal = X_test_anomaly
        test_data = np.concatenate((test_normal, test_abnormal), axis=0)
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        # Ground-truth labels for testing
        test_normal_labels = np.full((len(test_normal), 1), 0, dtype=int)
        test_abnormal_labels = np.full((len(test_abnormal), 1), 1, dtype=int)
        self.test_labels = np.concatenate(
            (test_normal_labels, test_abnormal_labels),
            axis=None,
        )

        logging.info(f"test data shape: {self.test.shape}")
        logging.info(f"train data shape: {self.train.shape}")
        logging.info(f"test_labels shape: {self.test_labels.shape}")


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class OpenStackSegLoader(object):

    def __init__(self, ensemble_param, data_path, win_size, step, data_seq_len, mode="train"):

        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Create preprocessor for loading data
        preprocessor = Preprocessor(
            length=data_seq_len,      # Extract sequences of length h
            timeout=float("inf"),     # No maximum allowed time between events
        )

        # Dataset paths
        path_train = data_path + "/train.txt"
        path_test_abnormal = data_path + "/test_abnormal.txt"
        path_test_normal = data_path + "/test_normal.txt"

        # Load data from OpenStack dataset
        X_train, _, _, _ = preprocessor.text(path_train, verbose=True)
        X_test, _, _, _ = preprocessor.text(path_test_normal, verbose=True)
        X_test_anomaly, _, _, _ = preprocessor.text(path_test_abnormal, verbose=True)

        # Fit scaler on training data
        data = X_train.numpy()
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Bootstrap sampling for training mode
        if self.mode == "train":
            random_number = get_random_state(
                "./model_config/bat_config/ensemble_train_os_config.yaml",
                ensemble_param[0],
                ensemble_param[1],
                ensemble_param[2],
                ensemble_param[3],
            )
            data, _ = resample(
                data,
                data,
                replace=True,
                n_samples=len(data),
                random_state=random_number,
            )

        # Prepare test data
        test_normal = X_test
        test_abnormal = X_test_anomaly
        test_data = np.concatenate((test_normal, test_abnormal), axis=0)
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        # Ground-truth labels for testing
        test_normal_labels = np.full((len(test_normal), 1), 0, dtype=int)
        test_abnormal_labels = np.full((len(test_abnormal), 1), 1, dtype=int)
        self.test_labels = np.concatenate(
            (test_normal_labels, test_abnormal_labels),
            axis=None,
        )

        logging.info(f"test data shape: {self.test.shape}")
        logging.info(f"train data shape: {self.train.shape}")
        logging.info(f"test_labels shape: {self.test_labels.shape}")

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



def get_loader_segment(ensemble_param, data_path, batch_size, win_size=100, step=100, data_seq_len=10, mode='train', dataset='BGL'):
    if (dataset == 'Openstack'):
        dataset = OpenStackSegLoader(ensemble_param, data_path, win_size, step, data_seq_len, mode)
    elif (dataset == 'HDFS'):
        dataset = HDFSSegLoader(ensemble_param, data_path, win_size, step, data_seq_len, mode)
    elif (dataset == 'BGL'):
        dataset = BGLSegLoader(ensemble_param, data_path, win_size, step, data_seq_len, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return data_loader
