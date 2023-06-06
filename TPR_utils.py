import torch
import torch.nn as nn
import torch.nn.functional as F
from node import Node

class TPR(nn.Module):
    def __init__(self, args, num_fillers, num_roles, d_filler=32, d_role=32, filler_emb_gain=1) -> None:
        super().__init__()
        self.filler_emb = nn.Embedding(num_fillers, d_filler) # +1 for <empty>
        self.role_emb = nn.Embedding(num_roles, d_role)
        self.learn_filler_embed = args.learn_filler_embed
        if not args.learn_filler_embed:
            self.filler_emb.requires_grad = False
            self.filler_emb.weight.requires_grad = False
        self.role_emb.requires_grad = False
        self.role_emb.weight.requires_grad = False
        # Attributes
        self.proj_filler_to_unit_ball = args.proj_filler_to_unit_ball
        self.filler_emb_gain = filler_emb_gain
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.learn_filler_embed:
            nn.init.xavier_uniform_(self.filler_emb.weight, gain=self.filler_emb_gain)
        else:
            nn.init.orthogonal_(self.filler_emb.weight, gain=self.filler_emb_gain)
            self.filler_emb.weight.data[0, :] = 0
        nn.init.orthogonal_(self.role_emb.weight, gain=1)
        self.filler_emb.weight.data[0, :] = 0

    def forward(self, tree_tensor):
        '''
        Given a binary tree represented by a tensor, construct the TPR
        '''
        if self.proj_filler_to_unit_ball:
            self.filler_emb.weight.data = self.filler_emb.weight.data / self.filler_emb.weight.data.norm(p=2, dim=-1).unsqueeze(1)
        x = self.filler_emb(tree_tensor)
        return torch.einsum('brm,rn->bmn', x, self.role_emb.weight)
    
    def unbind(self, tpr_tensor, decode=False):
        '''
        Given a TPR, unbind it
        '''
        unbinded = torch.einsum('bmn,rn->brm', tpr_tensor, self.role_emb.weight)
        if not decode:
            return unbinded
        return torch.einsum('brm,fm->brf', unbinded, self.filler_emb.weight)

@torch.no_grad()
def build_E(role_emb):
    '''
    Build E matrices given the role embeddings (binary trees-only)
    '''
    d_role = role_emb.weight.size(1)
    E_l = role_emb.weight.new_zeros(d_role, d_role)
    E_r = role_emb.weight.new_zeros(d_role, d_role)
    def _add_to(mat, ind_from, ind_to):
        if ind_to >= role_emb.weight.size(0):
            return
        mat += torch.einsum('a,b->ab', role_emb.weight[ind_to], role_emb.weight[ind_from])
        _add_to(mat, ind_from*2+1, ind_to*2+1)
        _add_to(mat, ind_from*2+2, ind_to*2+2)
    _add_to(E_l, 0, 1)
    _add_to(E_r, 0, 2)
    E_l.requires_grad = False
    E_r.requires_grad = False
    return E_l, E_r

@torch.no_grad()
def build_D(role_emb):
    '''
    Build D matrices given the role embeddings (binary trees-only)
    '''
    d_role = role_emb.weight.size(1)
    D_l = role_emb.weight.new_zeros(d_role, d_role)
    D_r = role_emb.weight.new_zeros(d_role, d_role)
    def _add_to(mat, ind_from, ind_to):
        if ind_from >= role_emb.weight.size(0):
            return
        mat += torch.einsum('a,b->ab', role_emb.weight[ind_to], role_emb.weight[ind_from])
        _add_to(mat, ind_from*2+1, ind_to*2+1)
        _add_to(mat, ind_from*2+2, ind_to*2+2)
    _add_to(D_l, 1, 0)
    _add_to(D_r, 2, 0)
    D_l.requires_grad = False
    D_r.requires_grad = False
    return D_l, D_r

def DecodedTPR2Tree(decoded_tpr, eps=1e-2):
    contain_symbols = decoded_tpr.norm(p=2, dim=-1) > eps
    return torch.where(contain_symbols, decoded_tpr.argmax(dim=-1), 0)

def gumbel_softmax(pi, t=1):
    u = torch.rand_like(pi)
    return F.softmax((-(-u.log()).log() + pi.log())*t, dim=-1)

# works for binary trees only
def Symbols2NodeTree(index_tree, i2v):
    def _traverse_and_detensorify(par, ind):
        if not index_tree[ind]:
            return par
        cur = Node(i2v[index_tree[ind]])
        if par:
            par.children.append(cur)
        if len(index_tree) > ind*2+1 and index_tree[ind*2+1]:
            # work on the left child
            _traverse_and_detensorify(cur, ind*2+1)
        if len(index_tree) > ind*2+2 and index_tree[ind*2+2]:
            # work on the right child
            _traverse_and_detensorify(cur, ind*2+2)
        return cur
    node_tree = _traverse_and_detensorify(None, 0)
    return node_tree

# example usage in main.py: BatchSymbols2NodeTree(fully_decoded, train_data.ind2vocab)
def BatchSymbols2NodeTree(decoded_tpr_batch, i2v):
    def s2nt(index_tree):
        return Symbols2NodeTree(index_tree, i2v)
    return list(map(s2nt, decoded_tpr_batch))
