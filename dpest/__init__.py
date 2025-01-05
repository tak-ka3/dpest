"""
確率質量関数
"""
class Pmf:
    def __init__(self, *args, val_to_prob=None):
        self.val_to_prob = val_to_prob or {}
        self.child = list(args)
        self.name = "Pmf"
        all_vars = set()
        self.is_args_depend = False # 引数間に依存関係があるか
        for child in self.child:
            depend_vars_set = set(child.depend_vars)
            if all_vars & depend_vars_set:
                self.is_args_depend = True
            all_vars.update(depend_vars_set)
        self.depend_vars = list(all_vars)
        self.is_sampling = False
