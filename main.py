from logging import warning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
import warnings
import argparse
import random
import time
import os
import re
import json
from utils import get_path_to_task_files
import math
from nltk import TreePrettyPrinter, Tree

try:
    from apex import amp
except BaseException as ex:
    warnings.warn('Apex is not installed. Half-precision training won\'t work.')

__has_wandb__ = False
try:
    import wandb
    # don't hardcode the key; get it from ENV VAR of client machine
    # for compute node, the key will be set by XT (from xt_config.yaml)
    if os.getenv("WANDB_API_KEY"):
        __has_wandb__ = True
except BaseException as ex:
    pass

from data import BinaryT2TDataset
from TPR_utils import TPR, DecodedTPR2Tree
from models import *

def calculate_loss(output, tpr, decoded, fully_decoded, entropies, args, logP_F=None):
    # Valid combinations are: [MSE, XENT], [MSE-onehot, empty], [XENT, empty]
    if args.tpr_loss_type == 'mse_oh_empty':
        # first-order loss (MSE-onehot)
        empty_positions = batch['output'] == 0
        tok_mse_loss = mse_loss(decoded[~empty_positions].view(-1, decoded.size(-1)),
                                F.one_hot(batch['output'][~empty_positions].view(-1),
                                          num_classes=decoded.size(-1)).float())
        # first-order <empty> token loss (mse)
        empty_loss = decoded[empty_positions].view(-1, decoded.size(-1)).norm(p=2, dim=-1)
        tpr_loss = (tok_mse_loss.sum() + args.tpr_loss_beta*empty_loss.sum()) / (tok_mse_loss.size(0) + empty_loss.size(0))

    elif args.tpr_loss_type == 'mse_xent':
        empty_positions = batch['output'] == 0
        # first-order loss (XENT)
        tok_xent_loss = xent_loss(decoded[~empty_positions].view(-1, decoded.size(-1)),
                                  batch['output'][~empty_positions].view(-1))
        # first-order loss (MSE)
        tpr_mse_loss = mse_loss(output, tpr(batch['output']).view(bsz, -1)).mean(-1)
        tpr_loss = (tpr_mse_loss.sum() + args.tpr_loss_beta*tok_xent_loss.sum()) / (tpr_mse_loss.size(0) + tok_xent_loss.size(0))
    elif args.tpr_loss_type == 'mse_oh_mse':
        tpr_loss = (tok_mse_loss.sum() + args.tpr_loss_beta*tpr_mse_loss.sum()) / (tok_mse_loss.size(0) + tpr_mse_loss.size(0))
    elif args.tpr_loss_type == 'xent_empty':
        tpr_loss = (tok_xent_loss.sum() + args.tpr_loss_beta*empty_loss.sum()) / (tok_xent_loss.size(0) + empty_loss.size(0))

    '''
    # zeroth-order loss
    # THIS IS ACTUALLY THE HVI LOSS!
    #if args.fp16:
    #    PG_loss = ((logP_F - ((fully_decoded==batch['output']).all(dim=-1).half()+1e-5).log())**2).mean()
    #else:
    #    PG_loss = ((logP_F - ((fully_decoded==batch['output']).all(dim=-1).float()+1e-5).log())**2).mean()
    # THIS IS THE ACTUAL PG LOSS
    if args.pg_reward_type == 'binary':
        reward = (fully_decoded==batch['output']).all(dim=-1)
    elif args.pg_reward_type == 'tpr':
        reward = (1 + cos_loss(output, tpr(batch['output']).view(bsz, -1)).detach()) / 2
        reward = reward ** (1/args.pg_reward_temp)
    else:
        raise NotImplementedError
    reward = reward.half() if args.fp16 else reward.float()

    if args.aux_loss_type == 'pg':
        reward = reward - reward.mean(0)
        aux_loss = -(logP_F * reward).mean()
    elif args.aux_loss_type == 'hvi':
        r = (logP_F - (reward+1e-8).log())
        r = r - r.mean(0)
        aux_loss = (r**2).mean()
    else:
        assert False

    if args.interleave_softmax and simple_agent.op_ge != 'softmax':
        total_loss = aux_loss
    else:
        if args.gamma >= 0:
            total_loss = tpr_loss + args.gamma * aux_loss
        else:
            total_loss = aux_loss
    '''
    total_loss = tpr_loss
    # exit earlier if loss is NaN
    if total_loss.isnan():
        exit(0)

    '''
    #writer.add_scalar('loss/primary', total_loss, step)
    pre_reg_loss = total_loss.clone().detach()    # make a copy of value (but not computation graph)

    operation_regularization = torch.tensor(0., device=device)
    arg_entropy_regularization = torch.tensor(0., device=device)

    if use_entropy_regularization:
        if args.use_loss_type_regularization:
            # here we flip entropy values to loss-additive certainty values (by substracting them from 1)

            # calc/apply operation_regularization
            operation_regularization_raw = (1-entropies['batch_per_layer_operation_entropy']).mean()
            operation_regularization = operation_entropy_regularization_coef * operation_regularization_raw
            total_loss = total_loss + operation_regularization

            # calc/apply arg_regularization
            arg_entropy_regularization_sum = (1-entropies['batch_per_layer_car_arg_entropy'].mean()) + \
                                         (1-entropies['batch_per_layer_cdr_arg_entropy'].mean()) + \
                                         (1-entropies['batch_per_layer_cons_arg1_entropy'].mean()) + \
                                         (1-entropies['batch_per_layer_cons_arg2_entropy'].mean())

            arg_entropy_regularization_raw = arg_entropy_regularization_sum/4
            arg_entropy_regularization = arg_entropy_regularization_coef * arg_entropy_regularization_raw 
            total_loss = total_loss + arg_entropy_regularization

        else:
            operation_regularization = operation_entropy_regularization_coef * entropies['batch_per_layer_operation_entropy'].mean()
            total_loss = total_loss - operation_regularization
            arg_entropy_regularization = entropies['batch_per_layer_car_arg_entropy'].mean() + \
                                            entropies['batch_per_layer_cdr_arg_entropy'].mean() + \
                                            entropies['batch_per_layer_cons_arg1_entropy'].mean() + \
                                            entropies['batch_per_layer_cons_arg2_entropy'].mean()
            arg_entropy_regularization =  arg_entropy_regularization_coef * arg_entropy_regularization 
            total_loss = total_loss - arg_entropy_regularization

    assert operation_regularization >= 0
    assert arg_entropy_regularization >= 0
    '''
    #return pre_reg_loss, operation_regularization, arg_entropy_regularization, total_loss
    return total_loss, torch.tensor(0., device=device), torch.tensor(0., device=device), total_loss

parser = argparse.ArgumentParser(description='pure TPR repo')
# arch
parser.add_argument('--d_filler', type=int, default=None,
                    help='dimension of filler vectors. Set to len(train_vocab) if None.')
parser.add_argument('--d_role', type=int, default=None,
                    help='dimension of role vectors. Set to 2**max_tree_depth if None.')
parser.add_argument('--d_key', type=int, default=64,
                    help='dimension of key vectors')
parser.add_argument('--blackboard_steps', type=int, default=3,
                    help='Number of steps to operate on the blackboard.')
parser.add_argument('--op_dist_fn', type=str, default='softmax', choices=['softmax', 'gumbel'],
                    help='The operation distribution function.')
parser.add_argument('--arg_dist_fn', type=str, default='softmax', choices=['softmax', 'gumbel'],
                    help='The argument distribution function.')
parser.add_argument('--per_layer_router', action='store_true',
                    help='Whether each layer should have it\'s own router')
parser.add_argument('--router_type', type=str, default='linear',
                    help='The transformation to use for the router. Options are [linear, mlp, gru, lstm, mixed_op_arg, set_transformer, enc_dec_transformer, enc_transformer, universal_transformer]')
parser.add_argument('--transformer_norm_first', type=int, default=1)
parser.add_argument('--transformer_activation', type=str, default='gelu')
parser.add_argument('--transformer_nheads', type=int, default=4)
parser.add_argument('--filler_emb_gain', type=float, default=1.0)
parser.add_argument('--input_norm', type=str, default=None, help='Whether to normalize the input. Options are [None, tpr_norm, ctrl_norm]')
parser.add_argument('--router_dropout', type=float, default=0.0,
                    help='Router dropout')
parser.add_argument('--router_hidden_dim', type=int, default=None,
                    help='Router hidden dim')
parser.add_argument('--router_num_layers', type=int, default=1,
                    help='Router num layers, this only applies to router_type [gru, lstm]')
parser.add_argument('--shared_keys', action='store_true',
                    help='Whether keys are shared across all agents on the blackboard.')
parser.add_argument('--proj_filler_to_unit_ball', action='store_true',
                    help='Whether to ensure that each filler vector has L2 norm 1')
parser.add_argument('--learn_filler_embed', action='store_true',
                    help='Whether to learn filler embeddings')
parser.add_argument('--ctrl_type', type=str, default='linear',
                    help='The transformation to use for the control state. Options are [linear, conv, conv_mlp]')
                    #help='The transformation to use for the control state. Options are [mean, mlp_mean_mlp, mlp_max_mlp, gru, conv, conv_mlp]')
parser.add_argument('--n_conv_kernels', type=int, default=4,
                    help='The number of output kernels to use for ctrl_type conv and conv_mlp.')
parser.add_argument('--ctrl_hidden_dim', type=int, default=64,
                    help='Ctrl hidden dim')
parser.add_argument('--ctrl_num_layers', type=int, default=1,)
parser.add_argument('--unrolled', type=float, default=0., help='If > 0, the ctrl_net and router will be unrolled '
                                                               'args.blackboard_steps times to help debug the gradient '
                                                               'flow through time steps. At the moment, this has'
                                                               'only been tested with transformer routers.')
parser.add_argument('--predefined_operations_are_random', action='store_true',
                    help='Whether the car/cdr/cons matrices are calculated exactly or learnable random matrices')
# training
parser.add_argument('--tpr_loss_type', default='mse_oh_empty',
                    choices=['mse_oh_empty', 'mse_xent', 'xent_empty', 'mse_oh_mse'])
parser.add_argument('--tpr_loss_beta', type=float, default=1.)
parser.add_argument('--epoch', type=int, default=None,
                    help='Number of training epochs. Either this or --steps must be set.')
parser.add_argument('--steps', type=float, default=None,
                    help='Number of training steps. Either this or --epoch must be set.')
parser.add_argument('--batch_size', type=int, default=8,
                    help='random seed')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--gs_temp', type=float, default=1.0,
                    help='gumbel softmax temperature')
parser.add_argument('--op_temp', type=float, default=1.0,
                    help='Operation softmax temperature')
parser.add_argument('--gamma', type=float, default=0,
                    help='loss function. -1 is REINFORCE, 0 is TPR, >0 is a blend.')
parser.add_argument('--aux_loss_type', default='pg', choices=['pg', 'hvi'])
parser.add_argument('--pg_reward_type', default='binary', choices=['binary', 'tpr'])
parser.add_argument('--pg_reward_temp', default=1., type=float)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--filler_map_location', default='None', choices=[None, 'pre_dtm', 'post_dtm', 'operation',
                                                                      'pre_shrink'])
parser.add_argument('--filler_map_type', default='linear', choices=['linear', 'mlp'])

parser.add_argument('--start_arg_temp', type=float, default=1.0, help='starting arg temp')
parser.add_argument('--end_arg_temp', type=float, default=1.0, help='ending arg temp')
parser.add_argument('--arg_temp_annealing_epochs', type=int, default=1, help='number of epochs to anneal arg temp')

parser.add_argument('--arg_entropy_regularization_start', type=float, default=0, help='starting argument entropy regularization coefficient overwrites entropy_regularization_start')
parser.add_argument('--operation_entropy_regularization_start', type=float, default=0, help='starting operation entropy regularization coefficient overwrites entropy_regularization_start')
parser.add_argument('--entropy_regularization_start', type=float, default=0, help='Starting entropy regularization for argument and operation')
parser.add_argument('--entropy_regularization_end', type=float, default=0.0, help='ending entropy regularization coefficient')
parser.add_argument('--entropy_regularization_epochs', type=float, default=0, help='The number of epochs to apply entropy regularization. [0 off, -1 args.epoch]')
parser.add_argument('--use_loss_type_regularization', type=int, default=1, help='When =1, use form of regularization appropriate for a loss term')

parser.add_argument('--eps_uniform', type=float, default=0.,
                    help='epsilon uniform for discrete exploration')
parser.add_argument('--softmax_steps', type=float, default=1,
                    help='how many softmax updates before switching to discrete')
parser.add_argument('--interleave_softmax', action='store_true',
                    help='Switch between op_ge and softmax')
# optim
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop', 'lamb'])
parser.add_argument('--optim_beta1', type=float, default=0.9)
parser.add_argument('--optim_beta2', type=float, default=0.98)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--scheduler', default='none', choices=['cosine', 'exponential', 'none'])
parser.add_argument('--scheduler_gamma', type=float, default=1, help='The gamma to apply if the scheduler is exponential.')
parser.add_argument('--wd', type=float, default=1e-2)
parser.add_argument('--gclip', type=float, default=1.)
parser.add_argument('--num_warmup_steps', type=float, default=0,
                    help='Number of warmup steps to linearly increase the learning rate from 0 to lr.')
parser.add_argument('--add_grad_noise', action='store_true')
parser.add_argument('--grad_noise_eta', type=float, default=.3)
parser.add_argument('--grad_noise_gamma', type=float, default=.55)
# IO
parser.add_argument('--max_tree_depth', type=int, default=6,
                    help='max depth of input trees')
parser.add_argument('--debug', type=int, default=1, help='1 [debug], 0 [off].')
parser.add_argument('--log_grad_norm', action='store_true',
                    help='Whether to log gradient norms during training.')
parser.add_argument('--save_file', type=str, default='result.tsv')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pt')
parser.add_argument('--task_path', type=str, default='nc_pat/v16/car_cdr_rcons', help='path from dataroot to task files')
parser.add_argument('--data_filter', type=str, default=None, help='Sequences to filter for in the dataset')
parser.add_argument('--max_train_examples', type=int, default=None, help='The number of training examples')
parser.add_argument('--validate_every_num_epochs', type=int, default=1,
                    help='How many training epochs do we wait before validating. The default is to validate after'
                         'every training epoch. This is useful when max_train_examples is set to a small number'
                         'so that we don\'t spend too much time validating.')
parser.add_argument('--wandb_tag', type=str, default='test')
parser.add_argument('--wandb_name', type=str, default=None)
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--train_log_freq', type=float, default=-1., help='training log frequency in steps, -1 logs at the end of each epoch')
args = parser.parse_args()
print(sorted(vars(args).items()))

if args.steps is None and args.epoch is None:
    assert False, 'Either --steps or --epoch must be set.'
elif args.steps is not None and args.epoch is not None:
    assert False, 'Only one of --steps or --epoch can be set.'

if args.router_hidden_dim is None:
    args.router_hidden_dim = args.ctrl_hidden_dim * 4

# wandb stuff
if __has_wandb__:
    wandb.init(project="TPR", entity=f"{os.environ.get('WANDB_USERNAME', default='edwardhu')}", config=args.__dict__, tags=[args.wandb_tag],
            mode="online" if args.use_wandb else "disabled", name=args.wandb_name)
    wandb.define_metric("train_acc", summary="max", step_metric='epoch')
    wandb.define_metric("valid_acc", summary="max", step_metric='epoch')
    wandb.define_metric("train_loss", summary="min", step_metric='epoch')
    wandb.define_metric("valid_loss", summary="min", step_metric='epoch')

# XT stuff
xt_run = None
if os.getenv("XT_RUN_NAME"):
    try:
        from xtlib.run import Run
        xt_run = Run()
        xt_run.log_hparams(args.__dict__)
    except ModuleNotFoundError:
        print('xtlib not found')

# TB stuff
writer = SummaryWriter()
writer.add_text('Hyperparameters', '\n'.join(str(x) for x in sorted(vars(args).items())), global_step=0)

# apply seed
if args.seed is None:
    # get a specific seed that we can report with the run
    args.seed = random.randint(0, 65535)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

batch_size = args.batch_size
max_depth = args.max_tree_depth
d_filler = args.d_filler
d_role = 2**max_depth if args.d_role is None else args.d_role
d_key = args.d_key

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# find/download data
if not os.getenv("HOME"):
    # provide default for HOME environment variable
    os.environ["HOME"] = os.path.expanduser("~/")

data_root = os.path.expandvars("$HOME/.data")
fn_xt_config = os.path.dirname(__file__) + "/xt_config.yaml"
task_path = get_path_to_task_files(data_root, args.task_path, fn_xt_config=fn_xt_config)

data_filter = re.compile(args.data_filter) if args.data_filter is not None else None
train_data = BinaryT2TDataset(os.path.join(task_path, 'train.xy'), max_depth=max_depth, device=device, filter=data_filter if data_filter else None,
                              max_examples=args.max_train_examples)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print('{} training examples'.format(len(train_data)))

if args.steps is not None:
    args.epoch = math.ceil(args.steps / len(train_loader))
    print('Steps set, training for {} epochs'.format(args.epoch))

valid_data = BinaryT2TDataset(os.path.join(task_path, 'dev.xy'), max_depth=max_depth, ind2vocab=train_data.ind2vocab, vocab2ind=train_data.vocab2ind, device=device, filter=data_filter if data_filter else None)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

test_data = BinaryT2TDataset(os.path.join(task_path, 'test.xy'), max_depth=max_depth, ind2vocab=train_data.ind2vocab, vocab2ind=train_data.vocab2ind, device=device, filter=data_filter if data_filter else None)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


eval_long_loader = None
eval_new_loader = None
if os.path.isfile(os.path.join(task_path, 'eval_long_adj.xy')):
    eval_long_data = BinaryT2TDataset(os.path.join(task_path, 'eval_long_adj.xy'), max_depth=max_depth,
                                      ind2vocab=train_data.ind2vocab, vocab2ind=train_data.vocab2ind,
                                      device=device, filter=data_filter if data_filter else None)
    eval_long_loader = DataLoader(eval_long_data, batch_size=batch_size, shuffle=False)

if os.path.isfile(os.path.join(task_path, 'eval_new_adj.xy')):
    eval_new_data = BinaryT2TDataset(os.path.join(task_path, 'eval_new_adj.xy'), max_depth=max_depth,
                                     ind2vocab=train_data.ind2vocab, vocab2ind=train_data.vocab2ind, device=device,
                                     filter=data_filter if data_filter else None)
    eval_new_loader = DataLoader(eval_new_data, batch_size=batch_size, shuffle=False)

if d_filler is None:
    d_filler = len(train_data.ind2vocab)

tpr = TPR(args, num_fillers=len(train_data.ind2vocab), num_roles=2**max_depth-1,
          d_filler=d_filler, d_role=d_role, filler_emb_gain=args.filler_emb_gain).to(device=device)

#simple_agent = nn.Sequential(nn.Linear(d_filler * d_role, d_filler * d_role)).cuda()

#simple_agent = MLPResNet(1, d_filler, d_role, rank=128).cuda()
#simple_agent = RouterNet(2, d_filler, d_role, rank=d_filler * d_role).cuda()
#simple_agent = KroneckerNet(d_filler, d_role).cuda()
#simple_agent = KroneckerRouterNet(2, d_filler, d_role).cuda()
# investigate why car/cdr aren't getting 100% val acc despite using a hard selection + having 100% train acc
#simple_agent = CarCdrConsNet(d_filler, d_role, D_l, D_r).cuda()
'''
simple_agent = FixedLenBB(d_filler, d_role, d_key, tpr.role_emb, length=args.blackboard_steps,
                          op_ge=args.op_grad_estimator, query_ge=args.query_grad_estimator, gs_temp=args.gs_temp,
                          ind2vocab=train_data.ind2vocab, tpr=tpr, per_layer_router=args.per_layer_router,
                          shared_keys=args.shared_keys, filler_map_type=args.filler_map_type,
                          eps_uniform=args.eps_uniform, interleave_softmax=args.interleave_softmax,
                          softmax_steps=args.softmax_steps, router_type=args.router_type,
                          router_dropout=args.router_dropout, router_hidden_dim=args.router_hidden_dim,
                          router_num_layers=args.router_num_layers, ctrl_type=args.ctrl_type,
                          ctrl_hidden_dim=args.ctrl_hidden_dim, ctrl_num_layers=args.ctrl_num_layers,
                          transformer_norm_first=args.transformer_norm_first, transformer_activation=args.transformer_activation,
                          input_norm=args.input_norm, unrolled=args.unrolled, transformer_nheads=args.transformer_nheads,
                          n_conv_kernels=args.n_conv_kernels).to(device=device)
'''
simple_agent = DiffTreeMachine(d_filler, d_role, args.ctrl_hidden_dim, tpr.role_emb, args.blackboard_steps,
                               args.ctrl_type, args.router_hidden_dim, nhead=args.transformer_nheads,
                               dropout=args.router_dropout, transformer_activation=args.transformer_activation,
                               transformer_norm_first=args.transformer_norm_first, op_dist_fn=args.op_dist_fn, arg_dist_fn=args.arg_dist_fn,
                               filler_map_type=args.filler_map_type, filler_map_location=args.filler_map_location,
                               n_conv_kernels=args.n_conv_kernels, ind2vocab=train_data.ind2vocab, tpr=tpr,
                               predefined_operations_are_random=args.predefined_operations_are_random).to(device=device)

if args.learn_filler_embed:
    params = chain(simple_agent.parameters(), [tpr.filler_emb.weight])
else:
    params = list(simple_agent.parameters())

trainable_params = list(filter(lambda p: p.requires_grad, params))
print('Trainable params: {}'.format(sum(p.numel() for p in trainable_params)))

if xt_run:
    xt_run.log_hparams({'parameters': sum(p.numel() for p in params),
                        'trainable_parameters': sum(p.numel() for p in trainable_params)})

if args.optimizer == 'adam':
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd, betas=(args.optim_beta1, args.optim_beta2),)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.wd, momentum=args.optim_beta1)
elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(trainable_params, lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'lamb':
    import torch_optimizer
    optimizer = torch_optimizer.Lamb(trainable_params, lr=args.lr, weight_decay=args.wd, betas=(args.optim_beta1, args.optim_beta2))

if args.scheduler == 'cosine':
    num_scheduler_epochs = args.epoch - math.ceil(args.num_warmup_steps / len(train_loader)) + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_scheduler_epochs, verbose=True)
elif args.scheduler == 'exponential':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)
else:
    scheduler = None

# apex magic
if args.fp16:
    (simple_agent, tpr), opt = amp.initialize(
        [simple_agent, tpr], optimizer,
        opt_level='O2')

mse_loss = torch.nn.MSELoss(reduction='none')
xent_loss = torch.nn.CrossEntropyLoss(reduction='none')
cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

torch.autograd.set_detect_anomaly(False)
best_train_acc = 0.
best_valid_acc = 0.
step = 0

use_entropy_regularization = args.entropy_regularization_epochs != 0
entropy_regularization_epochs = args.epoch if args.entropy_regularization_epochs == -1 else args.entropy_regularization_epochs
if args.entropy_regularization_start:
    if not args.arg_entropy_regularization_start:
        args.arg_entropy_regularization_start = args.entropy_regularization_start
    if not args.operation_entropy_regularization_start:
        args.operation_entropy_regularization_start = args.entropy_regularization_start

first_epoch_at_100 = None
lr = args.lr

iters = len(train_loader)
warmup_end_epoch = None
warmup_end_step = None

watch_gradients = False
if __has_wandb__ and watch_gradients:
    wandb.watch(simple_agent, log='gradients', log_freq=1)

# Log on the last training step of each epoch
if args.train_log_freq == -1:
    args.train_log_freq = iters - 1

for epoch_i in range(args.epoch):
    train_correct = 0
    partial_train_correct = 0
    partial_train_total = 0
    accum_i = 0
    train_total = 0
    train_pre_reg_loss_accum = 0
    train_op_reg_accum = 0
    train_arg_reg_accum = 0
    train_total_loss_accum = 0
    train_started = time.time()

    arg_temp = np.maximum(args.start_arg_temp - (args.start_arg_temp - args.end_arg_temp) * epoch_i / args.arg_temp_annealing_epochs, args.end_arg_temp)
    writer.add_scalar('Temperature/argument temp', arg_temp, step)

    arg_entropy_regularization_coef = args.arg_entropy_regularization_start - (args.arg_entropy_regularization_start - args.entropy_regularization_end) * epoch_i / entropy_regularization_epochs if use_entropy_regularization else 0
    arg_entropy_regularization_coef = np.maximum(arg_entropy_regularization_coef, 0.)
    writer.add_scalar('Regularization/arg entropy coefficient', arg_entropy_regularization_coef, step)

    operation_entropy_regularization_coef = args.operation_entropy_regularization_start - (
                args.operation_entropy_regularization_start - args.entropy_regularization_end) * epoch_i / entropy_regularization_epochs if use_entropy_regularization else 0
    operation_entropy_regularization_coef = np.maximum(operation_entropy_regularization_coef, 0.)
    writer.add_scalar('Regularization/operation entropy coefficient', operation_entropy_regularization_coef, step)

    gumbel_temp = 1

    # Training
    for i, batch in enumerate(train_loader):
        # On the last warmup step, the correct lr will be set
        is_warmup = step < args.num_warmup_steps
        if is_warmup:
            lr = args.lr * step / args.num_warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif warmup_end_epoch is None:
            warmup_end_epoch = epoch_i
            warmup_end_step = i

        if (args.op_dist_fn == 'gumbel' or args.arg_dist_fn == 'gumbel') and gumbel_temp > .5:
            gumbel_temp = max(.5, 1 - 1/args.steps * step)
            simple_agent.set_gumbel_temp(gumbel_temp)
            #print('Gumbel temp:', gumbel_temp)

        bsz = batch['input'].size(0)
        optimizer.zero_grad(set_to_none=True)
        # Use an agent
        output, _, entropies = simple_agent(tpr(batch['input']), calculate_entropy=use_entropy_regularization)
        # Use D_l directly
        #output = torch.einsum('bmn,cn->bmc', tpr(batch['input']), D_l)
        if args.fp16:
            decoded = tpr.unbind(output.half(), decode=True)
        else:
            decoded = tpr.unbind(output, decode=True)

        fully_decoded = DecodedTPR2Tree(decoded)

        # calc/accumulate loss
        pre_reg_loss, op_reg, arg_reg, total_loss = calculate_loss(output, tpr, decoded, fully_decoded, entropies, args)

        train_pre_reg_loss_accum += pre_reg_loss.item()
        train_op_reg_accum += op_reg.item() if type(op_reg) is torch.Tensor else op_reg
        train_arg_reg_accum += arg_reg.item() if type(arg_reg) is torch.Tensor else arg_reg
        train_total_loss_accum += total_loss.item()
        # calc/accumulate train acc
        train_correct += (fully_decoded == batch['output']).all(dim=-1).sum().item()
        train_total += batch['output'].size(0)

        empty_positions = batch['output']==0
        partial_train_correct += (fully_decoded == batch['output'])[~empty_positions].sum().item()
        # The denominator equals the number of non-empty positions plus the number of positions which should
        # be empty but are not
        partial_train_total += (~empty_positions).sum().item() + (fully_decoded != batch['output'])[empty_positions].sum().item()
        accum_i += 1

        # apex magic
        if args.fp16:
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        if args.log_grad_norm:
            parameters = simple_agent.named_parameters()
            parameters = [p for p in parameters if p[1].grad is not None]
            for parameter in parameters:
                writer.add_scalar('grad_norm/{}'.format(parameter[0]), parameter[1].grad.norm(), step)

        # TODO: look into the norm of our gradients, it seems very small
        if args.gclip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.gclip)
            else:
                torch.nn.utils.clip_grad_norm_(simple_agent.parameters(), args.gclip)

        if args.add_grad_noise:
            stddev = args.grad_noise_eta / ((1 + step) ** args.grad_noise_gamma)
            for p in simple_agent.parameters():
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad) * stddev
                p.grad.add_(noise)

        optimizer.step()
        # This commented out code is for the old cosine scheduler with warm restarts that is not currently used
        #if not is_warmup and args.scheduler == 'cosine':
        #    scheduler.step(epoch_i - warmup_end_epoch + i / iters)

        if step % args.train_log_freq == 0:
            train_pre_reg_loss = train_pre_reg_loss_accum / accum_i
            train_op_reg = train_op_reg_accum / accum_i
            train_arg_reg = train_arg_reg_accum / accum_i
            train_total_loss = train_total_loss_accum / accum_i
            writer.add_scalar('Loss/train', train_total_loss, step)
            writer.add_scalar('Loss/train_pre_reg', train_pre_reg_loss, step)
            writer.add_scalar('Loss/train_op_reg', train_op_reg, step)
            writer.add_scalar('Loss/train_arg_reg', train_arg_reg, step)
            train_acc = train_correct / train_total
            partial_train_acc = partial_train_correct / partial_train_total
            writer.add_scalar('Accuracy/train', train_acc, step)
            writer.add_scalar('Accuracy/partial_train', partial_train_acc, step)
            accum_i = 0
            train_correct = 0
            train_total = 0
            partial_train_correct = 0
            partial_train_total = 0
            train_pre_reg_loss_accum = 0
            train_op_reg_accum = 0
            train_arg_reg_accum = 0
            train_total_loss_accum = 0
            if __has_wandb__:
                wandb.log(dict(
                    epoch=epoch_i,
                    train_acc=train_acc,
                    partial_train_acc=partial_train_acc,
                    train_loss=train_total_loss,
                    pre_reg_loss=train_pre_reg_loss,
                    lr=lr,
                ))

        step += 1

        if args.steps and step > args.steps:
            break

    if epoch_i % args.validate_every_num_epochs == 0 or (args.steps and step == args.steps):
        # Evaluation
        with torch.inference_mode():
            simple_agent.eval()
            valid_loss_accum = 0
            valid_pre_reg_loss_accum = 0
            valid_correct = 0
            valid_total = 0
            partial_valid_correct = 0
            partial_valid_total = 0

            batch_average_operation_entropies_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)
            batch_average_car_arg_entropies_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)
            batch_average_cdr_arg_entropies_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)
            batch_average_cons_arg1_entropies_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)
            batch_average_cons_arg2_entropies_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)

            batch_average_operation_max_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)
            batch_average_car_arg_max_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)
            batch_average_cdr_arg_max_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)
            batch_average_cons_arg1_max_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)
            batch_average_cons_arg2_max_per_layer = torch.empty((len(valid_loader), args.blackboard_steps), device=device)

            for i, batch in enumerate(valid_loader):
                bsz = batch['input'].size(0)
                is_debug_step = i == 0 if args.debug else False
                output, debug_info, entropies = simple_agent(tpr(batch['input']), debug=is_debug_step,
                                                       calculate_entropy=False,)
                '''
                batch_average_operation_entropies_per_layer[i] = entropies['batch_per_layer_operation_entropy']
                batch_average_car_arg_entropies_per_layer[i] = entropies['batch_per_layer_car_arg_entropy']
                batch_average_cdr_arg_entropies_per_layer[i] = entropies['batch_per_layer_cdr_arg_entropy']
                batch_average_cons_arg1_entropies_per_layer[i] = entropies['batch_per_layer_cons_arg1_entropy']
                batch_average_cons_arg2_entropies_per_layer[i] = entropies['batch_per_layer_cons_arg2_entropy']
    
                batch_average_operation_max_per_layer[i] = debug_info['batch_per_layer_operation_arg_max']
                batch_average_car_arg_max_per_layer[i] = debug_info['batch_per_layer_car_arg_max']
                batch_average_cdr_arg_max_per_layer[i] = debug_info['batch_per_layer_cdr_arg_max']
                batch_average_cons_arg1_max_per_layer[i] = debug_info['batch_per_layer_cons_arg1_max']
                batch_average_cons_arg2_max_per_layer[i] = debug_info['batch_per_layer_cons_arg2_max']
                '''
                if is_debug_step:
                    formatted_tree = TreePrettyPrinter(Tree.fromstring(BatchSymbols2NodeTree(batch['output'], simple_agent.ind2vocab)[0].str()))
                    print('Correct output:\n{}'.format(formatted_tree.text()))
                    debug_text = debug_info['text']
                    writer.add_text('Epoch {}'.format(epoch_i), '\n\n'.join(debug_text), global_step=step)

                if args.fp16:
                    decoded = tpr.unbind(output.half(), decode=True)
                else:
                    decoded = tpr.unbind(output, decode=True)

                fully_decoded = DecodedTPR2Tree(decoded)

                pre_reg_loss, op_reg, arg_reg, total_valid_loss = calculate_loss(output, tpr, decoded, fully_decoded, entropies, args)

                valid_loss_accum += total_valid_loss.item()
                valid_pre_reg_loss_accum += pre_reg_loss.item()
                valid_correct += (fully_decoded==batch['output']).all(dim=-1).sum().item()
                valid_total += batch['output'].size(0)

                empty_positions = batch['output'] == 0
                partial_valid_correct += (fully_decoded == batch['output'])[~empty_positions].sum().item()
                # The denominator equals the number of non-empty positions plus the number of positions which should
                # be empty but are not
                partial_valid_total += (~empty_positions).sum().item() + (fully_decoded != batch['output'])[empty_positions].sum().item()

            simple_agent.train()

            valid_loss = valid_loss_accum / len(valid_loader)
            valid_pre_reg_loss = valid_pre_reg_loss_accum / len(valid_loader)

            valid_acc = valid_correct / valid_total
            partial_valid_acc = partial_valid_correct / partial_valid_total
            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(simple_agent.state_dict(), args.checkpoint_file)

            if args.debug:
                # log info to console at end of each epoch
                train_elapsed = time.time() - train_started
                train_rate = len(train_loader) / train_elapsed
                print(f'epoch: {epoch_i:,}')

                print(
                    f'  Train Acc: {train_acc:.2f}, partial_train_acc: {partial_train_acc:.2f}, total_loss: {train_total_loss:.5f}, lr: {lr:.10f}, samples/sec: {train_rate:.0f}, ' +
                    f'[pre_reg_loss: {train_pre_reg_loss:.5f}, ' +
                    # f'op_reg: {train_op_reg:.5f} ({operation_entropy_regularization_coef:.5f}*{train_op_reg/operation_entropy_regularization_coef:.5f}), ' +
                    # f'arg_reg: {train_arg_reg:.5f} ({arg_entropy_regularization_coef:.5f}*{train_arg_reg/arg_entropy_regularization_coef:.5f})]')
                    f'op_reg: {train_op_reg:.5f}, arg_reg: {train_arg_reg:.5f}]')

                print(
                    f'  Valid Acc: {valid_acc:.2f}, partial_valid_acc: {partial_valid_acc:.2f}, total_loss: {valid_loss:.5f}')

            # log TRAIN/VALID metrics to Tensorboard
            writer.add_scalar('LR', lr, step)
            writer.add_scalar('Accuracy/valid', valid_acc, step)
            writer.add_scalar('Accuracy/partial valid', partial_valid_acc, step)
            writer.add_scalar('Loss/valid', valid_loss, step)
            writer.add_scalar('Loss/valid_pre_reg', valid_pre_reg_loss, step)
            '''
            for entropy_i, average_entropy in enumerate(average_operation_entropy_per_layer):
                writer.add_scalar('Operation entropy/layer_{}'.format(entropy_i), average_entropy, step)
                writer.add_scalar('Car arg entropy/layer_{}'.format(entropy_i), average_car_arg_entropy_per_layer[entropy_i], step)
                writer.add_scalar('Cdr arg entropy/layer_{}'.format(entropy_i), average_cdr_arg_entropy_per_layer[entropy_i], step)
                writer.add_scalar('Cons arg1 entropy/layer_{}'.format(entropy_i), average_cons_arg1_entropy_per_layer[entropy_i], step)
                writer.add_scalar('Cons arg2 entropy/layer_{}'.format(entropy_i), average_cons_arg2_entropy_per_layer[entropy_i], step)

                writer.add_scalar('Operation max/layer_{}'.format(entropy_i), average_operation_max_per_layer[entropy_i], step)
                writer.add_scalar('Car arg max/layer_{}'.format(entropy_i), average_car_arg_max_per_layer[entropy_i], step)
                writer.add_scalar('Cdr arg max/layer_{}'.format(entropy_i), average_cdr_arg_max_per_layer[entropy_i], step)
                writer.add_scalar('Cons arg1 max/layer_{}'.format(entropy_i), average_cons_arg1_max_per_layer[entropy_i], step)
                writer.add_scalar('Cons arg2 max/layer_{}'.format(entropy_i), average_cons_arg2_max_per_layer[entropy_i], step)
            '''
            if xt_run:
                # log TRAIN/VALID metrics to XT
                md = {"epoch": int(epoch_i),  # JSON can't handle int64
                      "lr": lr,
                      "train_loss": train_total_loss,
                      "valid_loss": valid_loss,
                      "train_acc": train_acc,
                      "valid_acc": valid_acc,
                      "partial_train_acc": partial_train_acc,
                      "partial_valid_acc": partial_valid_acc,
                      "best_train_acc": best_train_acc,
                      "best_valid_acc": best_valid_acc,
                      "first_epoch_at_100": first_epoch_at_100,
                      "op_reg_coef": operation_entropy_regularization_coef.item(),
                      "arg_reg_coef": arg_entropy_regularization_coef.item(),
                      "pre_reg_loss": train_pre_reg_loss,
                      "op_reg": train_op_reg,
                      "arg_reg": train_arg_reg
                      }

                xt_run.log_metrics(md, step_name="epoch")

            if __has_wandb__:
                wandb.log(dict(
                    epoch=epoch_i,
                    valid_acc=valid_acc,
                    valid_loss=valid_loss,
                    valid_pre_reg_loss=valid_pre_reg_loss,
                ))


        '''
        average_operation_entropy_per_layer = batch_average_operation_entropies_per_layer.mean(axis=0)
        average_car_arg_entropy_per_layer = batch_average_car_arg_entropies_per_layer.mean(axis=0)
        average_cdr_arg_entropy_per_layer = batch_average_cdr_arg_entropies_per_layer.mean(axis=0)
        average_cons_arg1_entropy_per_layer = batch_average_cons_arg1_entropies_per_layer.mean(axis=0)
        average_cons_arg2_entropy_per_layer = batch_average_cons_arg2_entropies_per_layer.mean(axis=0)

        average_operation_max_per_layer = batch_average_operation_max_per_layer.mean(axis=0)
        average_car_arg_max_per_layer = batch_average_car_arg_max_per_layer.mean(axis=0)
        average_cdr_arg_max_per_layer = batch_average_cdr_arg_max_per_layer.mean(axis=0)
        average_cons_arg1_max_per_layer = batch_average_cons_arg1_max_per_layer.mean(axis=0)
        average_cons_arg2_max_per_layer = batch_average_cons_arg2_max_per_layer.mean(axis=0)
        '''

    best_train_acc = max(train_acc, best_train_acc)

    if best_valid_acc == 1.0 and not first_epoch_at_100:
        first_epoch_at_100 = epoch_i

    if scheduler and not is_warmup:
        # adjust LR as per scheduler
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

def calculate_accuracy(data_loader, data_name):
    with torch.inference_mode():
        correct = 0
        total = 0
        for i, batch in enumerate(data_loader):
            bsz = batch['input'].size(0)
            output, debug_info, _ = simple_agent(tpr(batch['input']), debug=False, calculate_entropy=False)

            if args.fp16:
                fully_decoded = DecodedTPR2Tree(tpr.unbind(output.half(), decode=True))
            else:
                fully_decoded = DecodedTPR2Tree(tpr.unbind(output, decode=True))

            correct += (fully_decoded == batch['output']).all(dim=-1).sum().item()
            total += batch['output'].size(0)

        print(f'{data_name} Acc: {correct / total:.2f}')
        writer.add_scalar('Accuracy/{}'.format(data_name), correct / total, step)

        if xt_run:
            md = {"epoch": epoch_i,
                  data_name: correct / total}
            xt_run.log_metrics(md, step_name="epoch")


# Reload the best checkpoint
simple_agent.load_state_dict(torch.load(args.checkpoint_file))

calculate_accuracy(test_loader, 'Test')
if eval_long_loader:
    calculate_accuracy(eval_long_loader, 'Eval_long_adj')

if eval_new_loader:
    calculate_accuracy(eval_new_loader, 'Eval_new_adj')

with open(args.save_file, 'a') as f:
    #f.write(f'{args.op_grad_estimator}\t{args.query_grad_estimator}\t{args.tpr_loss_type}\t{args.tpr_loss_beta}'
    #        f'\t{args.gs_temp}\t{args.seed}\t{args.gamma}\t{args.optimizer}\t{args.lr}'
    #        f'\t{train_correct/train_total:.3f}\t{valid_correct/valid_total:.3f}\n')
    '''
    log_obj = {'op_grad_estimator': args.op_grad_estimator,
               'query_grad_estimator': args.query_grad_estimator,
               'tpr_loss_type': args.tpr_loss_type,
               'tpr_loss_beta': args.tpr_loss_beta,
               'gs_temp': args.gs_temp,
               'seed': args.seed,
               'gamma': args.gamma,
               'optimizer': args.optimizer,
               'aux_loss_type': args.aux_loss_type,
               'beta1': args.optim_beta1,
               'beta2': args.optim_beta2,
               'lr': args.lr,
               'last_train_acc': train_correct/train_total,
               'last_valid_acc': valid_correct/valid_total,
               'best_train_acc': best_train_acc,
               'best_valid_acc': best_valid_acc}
    '''
    log_obj = {'last_train_acc': train_acc,
               'last_valid_acc': valid_correct/valid_total,
               'best_train_acc': best_train_acc,
               'best_valid_acc': best_valid_acc}
    log_obj.update(vars(args))
    f.write(json.dumps(log_obj)+'\n')

#print(f'op_ge : {args.op_grad_estimator}, query_ge : {args.query_grad_estimator}, gs_temp : {args.gs_temp},'
#      f'tpr_loss_type : {args.tpr_loss_type}, tpr_loss_beta : {args.tpr_loss_beta},'
#      f'seed : {args.seed}, gamma : {args.gamma}, optimizer : {args.optimizer}, lr : {args.lr}')
print("final metrics & hparams:\n{}\n".format(log_obj))

print(f'Best Train Acc: {best_train_acc:.3f}')
print(f'Best Valid Acc: {best_valid_acc:.3f}')
writer.close()
