class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children else []

    def get_max_depth(self):
        max_child_depth = 0

        for child in self.children:
            max_child_depth = max(max_child_depth, child.get_max_depth())

        return 1 + max_child_depth

    def get_max_branching(self):
        max_branching = 1

        if self.children:
            max_branching = len(self.children)

            for child in self.children:
                max_branching = max(max_branching, child.get_max_branching())

        return max_branching

    def get_num_nodes(self):
        node_count = 1

        for child in self.children:
            node_count += child.get_num_nodes()

        return node_count
    
    def __repr__(self, max_len=None, depth=0, add_parens=True):

        label_key = self.label

        if self.children:
            child_rep = ""

            for c, child in enumerate(self.children):

                flat_child = len(child.children) == 0
                child_text = child.__repr__(depth=depth+1, add_parens=(not flat_child))

                if max_len and depth==0 and len(child_text) > max_len:
                    child_text = child_text[0:max_len] + "..."
                if c:
                    child_rep += " " + child_text
                else:
                    child_rep += child_text
            rep = "{} {}".format(label_key, child_rep)

        else:
            # simple node without children
            rep = label_key

        if add_parens:
            rep = "( {} )".format(rep)

        return rep
    
    def __str__(self):
        return self.__repr__()

    def str(self, max_len=0):
        return self.__repr__(max_len)


def text_tree_to_node(tree):
    '''
    text tree conventions:
        ( A ) ==> parent A
        ( A B ) ==> parent A with child B
        ( A B C ) ==> parent A with children B and C
        ( A ( B C) ) ==> parent A with child B (B has child C)
    '''

    prev_tok = None
    stack = []
    parent = None
    for tok in tree.split():
        if prev_tok == '(':
            label = tok
            parent = Node(label)
            stack.append(parent) 
        elif tok == ')':
            child = parent 
            stack.pop()
            if len(stack) > 0:
                parent = stack[-1]
                parent.children.append(child)
        elif tok != '(':
            label = tok
            parent.children.append(Node(label))
        prev_tok = tok

    return parent