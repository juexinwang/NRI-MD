"""
Main training script for NRI-MD model.

Updated for Python 3.10+ and modern PyTorch compatibility.
"""

import time
import argparse
import pickle
import os
import datetime
from typing import Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from .utils import (
    load_dataset_train_valid_test, encode_onehot, get_triu_offdiag_indices, 
    get_tril_offdiag_indices, my_softmax, gumbel_softmax
)
from .modules import MLPEncoder, CNNEncoder, MLPDecoder, RNNDecoder, SimulationDecoder


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Neural relational inference for molecular dynamics simulations'
    )
    parser.add_argument('--num-residues', type=int, default=77,
                        help='Number of residues of the PDB.')
    parser.add_argument('--save-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    parser.add_argument('--load-folder', type=str, default='',
                        help='Where to load the trained model if finetunning. ' +
                             'Leave empty to train from scratch')
    parser.add_argument('--edge-types', type=int, default=4,
                        help='The number of edge types to infer.')
    parser.add_argument('--dims', type=int, default=6,
                        help='The number of input dimensions used in study( position (X,Y,Z) + velocity (X,Y,Z) ). ')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='The number of time steps per sample. Actually is 50')
    parser.add_argument('--prediction-steps', type=int, default=1, metavar='N',
                        help='Num steps to predict before re-using teacher forcing.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--encoder-hidden', type=int, default=256,
                        help='Number of hidden units in encoder.')
    parser.add_argument('--decoder-hidden', type=int, default=256,
                        help='Number of hidden units in decoder.')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')
    parser.add_argument('--encoder', type=str, default='mlp',
                        help='Type of path encoder model (mlp or cnn).')
    parser.add_argument('--decoder', type=str, default='rnn',
                        help='Type of decoder model (mlp, rnn, or sim).')
    parser.add_argument('--no-factor', action='store_true', default=False,
                        help='Disables factor graph model.')
    parser.add_argument('--encoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--decoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr-decay', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for data files.')
    parser.add_argument('--number-exp', type=int, default=1,
                        help='Number of experiments.')
    parser.add_argument('--number-expstart', type=int, default=0,
                        help='Starting experiment number.')
    parser.add_argument('--num-atoms', type=int, default=77,
                        help='Number of atoms (same as num-residues).')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--skip-first', action='store_true', default=False,
                        help='Skip first edge type in decoder, i.e. it represents no-edge.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='Uses discrete samples in training forward pass.')
    parser.add_argument('--prior', action='store_true', default=False,
                        help='Whether to use sparsity prior.')
    parser.add_argument('--dynamic-graph', action='store_true', default=False,
                        help='Whether test with dynamically re-computed graph.')

    return parser.parse_args()


def nll_gaussian(preds: torch.Tensor, target: torch.Tensor, variance: float, add_const: bool = False) -> torch.Tensor:
    """Negative log likelihood for Gaussian distribution."""
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def kl_categorical(preds: torch.Tensor, log_prior: torch.Tensor, num_atoms: int, eps: float = 1e-16) -> torch.Tensor:
    """KL divergence for categorical distribution."""
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds: torch.Tensor, num_atoms: int, num_edge_types: int, 
                          add_const: bool = False, eps: float = 1e-16) -> torch.Tensor:
    """KL divergence with uniform prior."""
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def edge_accuracy(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate edge prediction accuracy."""
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float64(correct) / (target.size(0) * target.size(1))


def train(epoch: int, best_val_loss: float, encoder: torch.nn.Module, decoder: torch.nn.Module,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
          train_loader: torch.utils.data.DataLoader, args: argparse.Namespace,
          rel_rec: torch.Tensor, rel_send: torch.Tensor, 
          log_prior: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """Train the model for one epoch."""
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    acc_train = []

    encoder.train()
    decoder.train()
    scheduler.step()

    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = data.float(), relations.long()

        optimizer.zero_grad()

        logits = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        prob = my_softmax(logits, -1)

        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec, rel_send, 1)
        else:
            output = decoder(data, edges, rel_rec, rel_send, args.prediction_steps)

        target = data[:, :, 1:, :]

        loss_nll = nll_gaussian(output, target, args.var)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        loss = loss_nll + loss_kl

        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)

        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'time: {:.4f}s'.format(time.time() - t))
    
    return np.mean(nll_val), np.mean(acc_val)


def test(encoder: torch.nn.Module, decoder: torch.nn.Module, 
         test_loader: torch.utils.data.DataLoader, args: argparse.Namespace,
         rel_rec: torch.Tensor, rel_send: torch.Tensor,
         log_prior: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """Test the model."""
    t = time.time()
    nll_test = []
    acc_test = []
    kl_test = []
    mse_test = []

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (data, relations) in enumerate(test_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()

            logits = encoder(data, rel_rec, rel_send)
            edges = gumbel_softmax(logits, tau=args.temp, hard=True)
            prob = my_softmax(logits, -1)

            # validation output uses teacher forcing
            output = decoder(data, edges, rel_rec, rel_send, 1)

            target = data[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

            acc = edge_accuracy(logits, relations)

            mse_test.append(F.mse_loss(output, target).item())
            nll_test.append(loss_nll.item())
            kl_test.append(loss_kl.item())
            acc_test.append(acc)

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)),
          'time: {:.4f}s'.format(time.time() - t))
    
    return np.mean(nll_test), np.mean(acc_test)


def main():
    """Main training function."""
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.factor = not args.no_factor
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Save model and meta-data
    if args.save_folder:
        exp_counter = 0
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')

        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'args': args}, open(meta_file, "wb"))
    else:
        print("WARNING: No save_folder provided!" +
              "Testing (within this script) will throw an error.")

    # Set num_atoms from num_residues if not set
    if not hasattr(args, 'num_atoms'):
        args.num_atoms = args.num_residues
    
    # Load data
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_dataset_train_valid_test(
        args.batch_size, args.number_exp, args.number_expstart, args.dims)

    # Generate off-diagonal interaction graph
    off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    if args.cuda:
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    # Create model
    if args.encoder == 'mlp':
        encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                           args.edge_types, args.encoder_dropout, args.factor)
    elif args.encoder == 'cnn':
        encoder = CNNEncoder(args.dims, args.encoder_hidden,
                           args.edge_types, args.encoder_dropout, args.factor)

    if args.decoder == 'mlp':
        decoder = MLPDecoder(n_in_node=args.dims,
                           edge_types=args.edge_types,
                           msg_hid=args.decoder_hidden,
                           msg_out=args.decoder_hidden,
                           n_hid=args.decoder_hidden,
                           do_prob=args.decoder_dropout,
                           skip_first=args.skip_first)
    elif args.decoder == 'rnn':
        decoder = RNNDecoder(n_in_node=args.dims,
                           edge_types=args.edge_types,
                           n_hid=args.decoder_hidden,
                           do_prob=args.decoder_dropout,
                           skip_first=args.skip_first)
    elif args.decoder == 'sim':
        decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

    if args.load_folder:
        encoder_file = os.path.join(args.load_folder, 'encoder.pt')
        encoder.load_state_dict(torch.load(encoder_file))
        decoder_file = os.path.join(args.load_folder, 'decoder.pt')
        decoder.load_state_dict(torch.load(decoder_file))

        args.save_folder = False

    if args.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                          lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                   gamma=args.gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(args.num_atoms)
    tril_indices = get_tril_offdiag_indices(args.num_atoms)

    if args.prior:
        prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
        print("Using prior")
        print(prior)
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = log_prior.repeat(args.batch_size, triu_indices.size(0), 1)
        if args.cuda:
            log_prior = log_prior.cuda()

    else:
        log_prior = None

    # Train model
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss, acc = train(epoch, best_val_loss, encoder, decoder, optimizer,
                             scheduler, train_loader, args, rel_rec, rel_send, log_prior)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if args.save_folder:
                torch.save(encoder.state_dict(), encoder_file)
                torch.save(decoder.state_dict(), decoder_file)
        print("Best Epoch: {:04d}".format(best_epoch),
              "Best Val Loss: {:.10f}".format(best_val_loss))

        if epoch % 10 == 0:
            test(encoder, decoder, test_loader, args, rel_rec, rel_send, log_prior)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model for testing
    if args.save_folder:
        encoder.load_state_dict(torch.load(encoder_file))
        decoder.load_state_dict(torch.load(decoder_file))

    test_loss, test_acc = test(encoder, decoder, test_loader, args, rel_rec, rel_send, log_prior)

    if args.save_folder:
        print("Best Val Loss: {:.10f}".format(best_val_loss), file=log)
        print("Test Loss: {:.10f}".format(test_loss), file=log)
        print("Test Acc: {:.10f}".format(test_acc), file=log)
        log.close()


if __name__ == '__main__':
    main()