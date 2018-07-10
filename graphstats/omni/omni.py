def omni_matrix(list_of_sim_matrices, off_diag = "mean"):
    """
    Inputs
        list_of_sim_matrices - The adjacencies to create the omni for
        off_diag = Metric used for off diagonals

    Returns
        omni - The omni matrix of the list
    """

    M = len(list_of_sim_matrices)
    n = len(list_of_sim_matrices[0])
    omni = np.zeros(shape = (M*n, M*n))

    for i in range(M):
        for j in range(i, M):
            for k in range(n):
                for m in range(k + 1, n):
                    if i == j:
                        omni[i*n + k, j*n + m] = list_of_sim_matrices[i][k, m] 
                        omni[j*n + m, i*n + k] = list_of_sim_matrices[i][k, m] # symmetric
                    else:
                        if off_diag == "mean":
                            omni[i*n + k, j*n + m] = (list_of_sim_matrices[i][k,m] + list_of_sim_matrices[j][k,m])/2
                            omni[j*n + m, i*n + k] = (list_of_sim_matrices[i][k,m] + list_of_sim_matrices[j][k,m])/2
    return omni