def input_generator(adj, size=5):
    """
    テストケースの生成
    """
    input_list = []
    if size == 5:
        input_list.append(([1, 1, 1, 1, 1], [2, 1, 1, 1, 1]))
        input_list.append(([1, 1, 1, 1, 1], [0, 1, 1, 1, 1]))
        if adj == "1":
            return input_list
        elif adj == "inf":
            input_list.append(([1, 1, 1, 1, 1], [2, 0, 0, 0, 0]))
            input_list.append(([1, 1, 1, 1, 1], [0, 2, 2, 2, 2]))
            input_list.append(([1, 1, 1, 1, 1], [0, 0, 0, 2, 2]))
            input_list.append(([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]))
            input_list.append(([1, 1, 0, 0, 0], [0, 0, 1, 1, 1]))
            return input_list
        else:
            raise ValueError("Invalid adj value")
    elif size == 10:
        input_list.append(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        input_list.append(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        if adj == "1":
            return input_list
        elif adj == "inf":
            input_list.append(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            input_list.append(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
            input_list.append(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]))
            input_list.append(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
            input_list.append(([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
            return input_list
    else:
        raise ValueError("Invalid size value")
