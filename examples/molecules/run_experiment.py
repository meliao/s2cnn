# pylint: disable=E1101,R,C
import argparse
import logging
from typing import Dict
import torch
import torch.nn as nn
from torch.autograd import Variable
from s2cnn_model import S2CNNRegressor
from baseline_model import BaselineRegressor
from utils import load_data, IndexBatcher, to_one_hot, exp_lr_scheduler, \
        count_params
import numpy as np
from timeit import default_timer


OPTIMIZER = torch.optim.Adam

NUM_ATOM = 23
NUM_ATOM_TYPES = 6


def eval_batch_mlp(mlp, data, batch_idxs, criterion, device_id=0):
    """ evaluate a batch for the baseline mlp """
    atom_types = to_one_hot(data["features"]["atom_types"][batch_idxs, ...],
                            NUM_ATOM_TYPES)
    targets = data["targets"][batch_idxs, ...]

    atom_types = Variable(atom_types)
    targets = Variable(targets)

    if torch.cuda.is_available():
        atom_types = atom_types.cuda(device_id)
        targets = targets.cuda(device_id)

    outputs = mlp(atom_types)
    loss = criterion(outputs, targets)
    return loss


def eval_batch_s2cnn(mlp, s2cnn, data, batch_idxs, criterion, device_id=0):
    """ evaluate a batch for the s2cnn """
    geometry = data["features"]["geometry"][batch_idxs, ...]
    atom_types = data["features"]["atom_types"][batch_idxs, ...]
    atom_types_one_hot = to_one_hot(atom_types, NUM_ATOM_TYPES)
    targets = data["targets"][batch_idxs, ...]

    geometry = Variable(geometry)
    atom_types = Variable(atom_types)
    atom_types_one_hot = Variable(atom_types_one_hot)
    targets = Variable(targets)

    if torch.cuda.is_available():
        atom_types_one_hot = atom_types_one_hot.cuda(device_id)
        geometry = geometry.cuda(device_id)
        atom_types = atom_types.cuda(device_id)
        targets = targets.cuda(device_id)

    outputs = mlp(atom_types_one_hot)
    outputs += s2cnn(geometry, atom_types)

    loss = criterion(outputs, targets)

    return loss


def train_baseline(mlp, data, train_batches, test_batches, num_epochs,
                   learning_rate_mlp, device_id=0):
    """ train the baseline model """
    optim = OPTIMIZER(mlp.parameters(), lr=learning_rate_mlp)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda(device_id)
    for epoch in range(num_epochs):
        train_losses = []
        # print("training")
        for iteration, batch_idxs in enumerate(train_batches):
            mlp.train()
            optim.zero_grad()
            loss = eval_batch_mlp(mlp, data, batch_idxs, criterion, device_id)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
            # print("\riteration {}/{}".format(
            #     iteration+1, train_batches.num_iterations()), end="")
        # print()
        test_losses = []
        # print("evaluating")
        for iteration, batch_idxs in enumerate(test_batches):
            mlp.eval()
            loss = eval_batch_mlp(mlp, data, batch_idxs, criterion)
            test_losses.append(loss.item())
            # print("\riteration {}/{}".format(
            #     iteration+1, test_batches.num_iterations()), end="")
        # print()
        train_loss = np.sqrt(np.mean(train_losses))
        test_loss = np.sqrt(np.mean(test_losses))
        # test_MAE = np.mean(np.abs(test_losses))

        logging.info("epoch {}/{} - avg train loss: {}, test loss: {}".format(
            epoch+1, num_epochs, train_loss, test_loss))
    return train_loss, test_loss


def train_s2cnn(mlp, s2cnn, data, train_batches, test_batches, num_epochs,
                init_learning_rate_s2cnn, learning_rate_decay_epochs,
                device_id=0):
    """ train the s2cnn keeping the baseline frozen """
    optim = OPTIMIZER(s2cnn.parameters(), lr=init_learning_rate_s2cnn)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda(device_id)
    for epoch in range(num_epochs):
        optim = exp_lr_scheduler(optim, epoch,
                                 init_lr=init_learning_rate_s2cnn,
                                 lr_decay_epoch=learning_rate_decay_epochs)
        train_losses = []
        # print("training")
        for iteration, batch_idxs in enumerate(train_batches):
            s2cnn.train()
            mlp.eval()
            optim.zero_grad()
            loss = eval_batch_s2cnn(mlp, s2cnn, data, batch_idxs, criterion)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        #     print("\riteration {}/{} - batch loss: {}".format(
        #         iteration+1, train_batches.num_iterations(),
        #         np.sqrt(train_losses[-1])), end="")
        # print()
        test_losses = []
        # print("evaluating")
        for iteration, batch_idxs in enumerate(test_batches):
            s2cnn.eval()
            mlp.eval()
            loss = eval_batch_s2cnn(mlp, s2cnn, data, batch_idxs, criterion)
            test_losses.append(loss.item())
        #     print("\riteration {}/{}  - batch loss: {}".format(
        #         iteration+1, test_batches.num_iterations(),
        #         np.sqrt(test_losses[-1])), end="")
        # print()
        train_loss = np.sqrt(np.mean(train_losses))
        test_loss = np.sqrt(np.mean(test_losses))
        # test_MAE = np.mean(np.abs(test_losses))
        logging.info("epoch {}/{} - avg train loss: {}, test loss: {}".format(
            epoch+1, num_epochs, train_loss, test_loss))
    return train_loss, test_loss

def evaluate_test_MAE(mlp_model: BaselineRegressor, 
                        s2cnn_model: S2CNNRegressor, 
                        data: Dict, 
                        test_batches: IndexBatcher,
                        device_id: int) -> None:
    all_targets = []
    all_outputs = []
    logging.info("Computing test MAE. test_batches has shape: %s", test_batches.n_batch)

    for batch_idxs in test_batches:    
        geometry = data["features"]["geometry"][batch_idxs, ...]
        atom_types = data["features"]["atom_types"][batch_idxs, ...]
        atom_types_one_hot = to_one_hot(atom_types, NUM_ATOM_TYPES)
        targets = data["targets"][batch_idxs, ...]
        all_targets.append(targets.numpy())
        geometry = Variable(geometry)
        atom_types = Variable(atom_types)
        atom_types_one_hot = Variable(atom_types_one_hot)
        targets = Variable(targets)

            
        if torch.cuda.is_available():
            atom_types_one_hot = atom_types_one_hot.cuda(device_id)
            geometry = geometry.cuda(device_id)
            atom_types = atom_types.cuda(device_id)
            targets = targets.cuda(device_id)

        outputs = mlp_model(atom_types_one_hot)
        outputs += s2cnn_model(geometry, atom_types)
        all_outputs.append(outputs.cpu().detach().numpy())

    all_targets = np.array(all_targets).flatten()
    all_outputs = np.array(all_outputs).flatten()

    MAE = np.mean(np.abs(all_targets - all_outputs))
    logging.info("Test MAE: %f", MAE)
    # logging.info("mlp_model: %s, s2cnn_model: %s, data: %s, test_batches: %s",
    #                 type(mlp_model),
    #                 type(s2cnn_model),
    #                 type(data),
    #                 type(test_batches))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        type=str,
                        default="data.joblib")
    parser.add_argument("--test_strat",
                        type=int,
                        default=0)
    parser.add_argument("--device_id",
                        type=int,
                        default=0)
    parser.add_argument("--num_epochs_s2cnn",
                        type=int,
                        default=30)
    parser.add_argument("--num_epochs_mlp",
                        type=int,
                        default=30)
    parser.add_argument("--batch_size_s2cnn",
                        type=int,
                        default=32)
    parser.add_argument("--batch_size_mlp",
                        type=int,
                        default=32)
    parser.add_argument("--init_learning_rate_s2cnn",
                        type=int,
                        default=1e-3)
    parser.add_argument("--learning_rate_mlp",
                        type=int,
                        default=1e-3)
    parser.add_argument("--learning_rate_decay_epochs",
                        type=int,
                        default=10)
    parser.add_argument('--log_file', type=str)

    args = parser.parse_args()
    fmt = "%(asctime)s:s2cnn: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S' 
    logging.basicConfig(filename=args.log_file,
                        filemode='a',
                        level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)

    torch.cuda.set_device(args.device_id)

    logging.info("evaluating on {}".format(args.test_strat))

    # logging.info("loading data...", end="")
    logging.info("Loading data from %s", args.data_path)
    data, train_idxs, test_idxs = load_data(args.data_path, args.test_strat,
                                            cuda=args.device_id)

    mlp = BaselineRegressor()
    s2cnn = S2CNNRegressor()

    if torch.cuda.is_available():
        for model in [mlp, s2cnn]:
            model.cuda(args.device_id)

    logging.info("training baseline MLP model")
    logging.info("mlp #params: {}".format(count_params(mlp)))
    t1 = default_timer()
    train_baseline(mlp, data,
                   IndexBatcher(train_idxs, args.batch_size_mlp,
                                cuda=args.device_id),
                   IndexBatcher(test_idxs, args.batch_size_mlp,
                                cuda=args.device_id),
                   args.num_epochs_mlp, args.learning_rate_mlp, args.device_id)

    logging.info("training residual s2cnn model")
    logging.info("s2cnn #params: {}".format(count_params(s2cnn)))
    train_s2cnn(mlp, s2cnn, data,
                IndexBatcher(train_idxs, args.batch_size_s2cnn,
                             cuda=args.device_id),
                IndexBatcher(test_idxs, args.batch_size_s2cnn,
                             cuda=args.device_id),
                args.num_epochs_s2cnn, args.init_learning_rate_s2cnn,
                args.learning_rate_decay_epochs, args.device_id)

    test_batches = IndexBatcher(test_idxs, args.batch_size_s2cnn,
                             cuda=args.device_id)
    t2 = default_timer()
    logging.info("Total training time %f", t2 - t1)
    evaluate_test_MAE(mlp, s2cnn, data, test_batches, args.device_id)

if __name__ == '__main__':

    main()
