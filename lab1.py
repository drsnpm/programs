def aStarAlgo(start_node, stop_node):
    open_set = set([start_node])
    closed_set = set()
    g = {}
    parents = {}
    
    g[start_node] = 0
    parents[start_node] = start_node
    
    while len(open_set) > 0:
        n = None

        for v in open_set:
            if n is None or g[v] + h(v) < g[n] + h(n):
                n = v
        
        if n is None:
            print('Path does not exist!')
            return None
        
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path
        
        for (m, weight) in get_neighbors(n):
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

        open_set.remove(n)
        closed_set.add(n)
    
    print('Path does not exist!')
    return None

Graph_nodes = {
    'S': [('A', 1), ('B', 2)],
    'A': [('E', 13)],
    'B': [('E', 5)]
}

def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return []

def h(n):
    H_dist = {
        'S': 5,
        'A': 4,
        'B': 5,
        'E': 0
    }
    return H_dist.get(n, float('inf'))

aStarAlgo('S', 'E')
