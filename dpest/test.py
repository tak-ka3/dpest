class Base:
    def __init__(self, name, is_depend=False, child=None):
        """
        クラスの基底構造を表現
        Args:
            name (str): クラスの名前
            is_depend (bool): True の場合、Pmf に置き換える
            child (list): 子インスタンスのリスト
        """
        self.name = name
        self.is_depend = is_depend
        self.child = child if child else []  # 子インスタンスのリスト

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, is_depend={self.is_depend})"


class Laplace(Base):
    pass  # Laplaceインスタンスを示すクラス


class Pmf(Base):
    def __init__(self, original_instance):
        """
        Pmfクラスのコンストラクタ
        Args:
            original_instance (Base): 置き換えられる元のインスタンス
        """
        super().__init__(name=f"Pmf({original_instance.name})", is_depend=False)
        self.original_instance = original_instance


def traverse_and_replace(node):
    """
    木構造を走査し、Laplaceインスタンスまで遡る。
    `is_depend` が True の場合にノードを Pmf に置き換える。
    Args:
        node (Base): 現在のノード
    Returns:
        Base: 更新されたノード
    """
    if isinstance(node, Laplace):
        # Laplaceインスタンスの場合、処理を終了
        return node

    if node.is_depend:
        # is_depend が True の場合、Pmfインスタンスに置き換える
        print(f"Replacing {node} with Pmf instance.")
        return Pmf(node)

    # 子ノードを再帰的に処理
    updated_children = [traverse_and_replace(child) for child in node.child]
    node.child = updated_children
    return node


# 木構造の例
root = Base("root", is_depend=False, child=[
    Base("child1", is_depend=True),
    Base("child2", is_depend=False, child=[
        Laplace("laplace1"),
        Base("child3", is_depend=True)
    ])
])

print("Before:")
print(root)

# 木構造を走査して置き換え
updated_root = traverse_and_replace(root)

print("\nAfter:")
print(updated_root.child)
