import random
import torch
from torch.utils.data import Dataset

from node import text_tree_to_node

class BinaryT2TDataset(Dataset):
    '''
    Trees are represented as vectors of indices of length 2**depth
    '''

    def __init__(self, tsv_file, max_examples=None, filter=None, max_depth=20, ind2vocab=None, vocab2ind=None, device='cuda'):
        self.device = device
        self.max_depth = max_depth
        if ind2vocab is None:
            self.ind2vocab = ['<empty>']
            self.vocab2ind = {'<empty>': 0}
        else:
            self.ind2vocab = ind2vocab
            self.vocab2ind = vocab2ind

        with open(tsv_file) as f:
            rows = list(f)

            if filter is not None:
                def filter_match(row, dfilter):
                    field3 = row.split("\t")[2].strip()
                    return dfilter.search(field3)

                rows = [row for row in rows if filter_match(row, filter)]

            if max_examples:
                random.shuffle(rows)
                rows = rows[0:max_examples]

            #print("data rows loaded: {:}".format(len(rows)))

            self.data = self.process_trees(rows, print_max_depth=max_depth==0)


    def process_trees(self, data, print_max_depth=False):
        processed = []
        max_branch = 0

        dataset_max_depth = 0

        for line in data:
            inout_pair = line.split('\t')
            tt = None

            if len(inout_pair) > 2:
                # remove 3rd field used for filtering
                tt = inout_pair[2].strip()
                inout_pair = inout_pair[0:2]  

            in_node, out_node = list(map(text_tree_to_node, inout_pair))
            example = {"input": in_node, "output": out_node, "example_type": tt}

            max_branch = max([max_branch, in_node.get_max_branching(), out_node.get_max_branching()])
            assert max_branch <= 2

            if print_max_depth and example['input'].get_max_depth() > dataset_max_depth:
                dataset_max_depth = example['input'].get_max_depth()
            if print_max_depth and example['output'].get_max_depth() > dataset_max_depth:
                dataset_max_depth = example['output'].get_max_depth()

            if example['input'].get_max_depth() > self.max_depth or example['output'].get_max_depth() > self.max_depth:
                continue
            
            # add to vocab
            def _add_to_vocab(node):
                if node is None:
                    return
                if node.label not in self.vocab2ind:
                    self.ind2vocab.append(node.label)
                    self.vocab2ind[node.label] = len(self.ind2vocab) - 1
                for child in node.children:
                    _add_to_vocab(child)
                return
            
            _add_to_vocab(example['input'])
            _add_to_vocab(example['output'])

            processed.append(example)

        if print_max_depth:
            print('Max depth seen in dataset: {}'.format(dataset_max_depth))
        return processed

    def get_direct(self, idx):
        return self.__getitem__(idx, apply_transforms=False)
        
    def __getitem__(self, idx, apply_transforms=True):
        '''
        Gets the specified text input/output and converts it to a set of tensors.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get input/output node trees as dict
        sample = self.data[idx]

        if apply_transforms:
            transformed_sample = {}
            transformed_sample['example_type'] = sample['example_type'] if sample['example_type'] is not None else 0
            transformed_sample['input'] = self.text_to_tensors(sample['input'])
            transformed_sample['output'] = self.text_to_tensors(sample['output'])
            return transformed_sample

        return sample
    
    def text_to_tensors(self, node):
        sample_tensor = torch.zeros((2**self.max_depth-1, ), device=self.device, dtype=torch.long)
        def _traverse_and_tensorify(node, ind):
            if node is None:
                return
            sample_tensor[ind] = self.vocab2ind[node.label]
            if len(node.children) > 0:
                # work on the left child
                _traverse_and_tensorify(node.children[0], ind*2+1)
            if len(node.children) > 1:
                # work on the right child
                _traverse_and_tensorify(node.children[1], ind*2+2)
            return
        _traverse_and_tensorify(node, 0)
        return sample_tensor

    def __len__(self):
        return len(self.data)