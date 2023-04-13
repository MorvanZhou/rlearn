import pathfind


def get_graph(maze):
    _COST_MAP = {
        1: pathfind.INFINITY,
        0: 1,
        None: pathfind.INFINITY,
    }
    matrix = []
    for row in maze:
        new_row = []
        for cell in row:
            cost = _COST_MAP[cell]
            new_row.append(cost)
        matrix.append(new_row)
    return pathfind.transform.matrix2graph(matrix, diagonal=False)


def move(nr, nc, mr, mc):
    if nr == mr:
        if nc > mc:
            return "R"
        else:
            return "L"
    else:
        dr = nr - mr
        if dr == 1:
            return "D"
        elif dr == 0:
            return "S"
        else:
            return "U"


def short_gem(me, items, graph):
    short_path = None
    short_count = 9999999
    for k, v in items.items():
        if k.endswith("_gem"):
            item = v[0]
            row = item.row
            col = item.col
            path = pathfind.find(graph, start=f"{me.row},{me.col}", end=f"{row},{col}")
            if short_path is None:
                short_path = path
            new_count = len(path)
            if short_count > new_count:
                short_count = new_count
                short_path = path
    return short_path


def _go(path, me):
    nr, nc = path[1].split(",")
    a = move(int(nr), int(nc), me.row, me.col)
    return a


def go(me, exit, items, graph):
    e_path = pathfind.find(graph, start=f"{me.row},{me.col}", end=f"{exit.row},{exit.col}")
    g_path = short_gem(me, items, graph)
    if len(e_path) >= me.energy:
        return _go(e_path, me)
    else:
        return _go(g_path, me)
