import newick as nwk

# function to convert `*.nwk` trees into numpy matrices of edge-lengths
def nwk_read(nwk_filename, leaves=None):

    root = nwk.read(nwk_filename)[0]

    # first, add all nodes and edges into arrays
    nodes = []
    edges = []

    def parse_subtree(node):
        nonlocal nodes
        nonlocal edges

        if not node in nodes:
            nodes = nodes + [node]
            for child in node.descendants:
                parse_subtree(child)
                edges = edges + [(nodes.index(child),nodes.index(node), child.length)]

    parse_subtree(root)
    
    # use the edges to construct an edge-length matrix
    n = len(nodes)
    T = np.zeros((n, n))

    for edge in edges:
        i = edge[0]
        j = edge[1]
        length = edge[2]
        T[i,j] = length
        
    # now, if leave names are given, ensure the leaves end
    # up in the resepective columns of the matrix
    if leaves != None:
        names = [node.name for node in nodes]

        for i in range(len(leaves)):
            leave = leaves[i]
            j = names.index(leave)

            T[[i,j]] = T[[j,i]]
            T[:,[i,j]] = T[:,[j,i]]

    # ensure an upper tridiagonal structure of the matrix
    ##### this is simply done by bubble sort. if big graphs are to 
    # be considered, one should use a better sorting algortihm ######
    s = 0
    if leaves != None:
        s = len(leaves)
    
    for i in range(s,n):
        for j in range(n-i):
            if T[i+j,i] > 0:
                T[[i+j,i]] = T[[i,i+j]]
                T[:,[i+j,i]] = T[:,[i,i+j]]
                
            
    return T
