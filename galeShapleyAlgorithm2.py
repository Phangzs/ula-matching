import numpy as np
from scipy.io import loadmat

def combine_by_summing(arr, groups, axis=0):
    groups = np.array(groups, dtype=int)
    
    # Check that the groups sum to the size along the given axis.
    if groups.sum() != arr.shape[axis]: exit(1)
    
    # Compute the starting indices of each group.
    indices = np.concatenate(([0], np.cumsum(groups)[:-1]))
    
    # Use np.add.reduceat to sum over the specified groups.
    return np.add.reduceat(arr, indices, axis=axis)

def GaleShapleyAlgorithmQuota(P1, P2, quota):
    m, n = P1.shape  # m = size of Group 2, n = size of Group 1

    # Convert from 1-indexed to 0-indexed
    P1 = P1 - np.ones_like(P1)
    P2 = P2 - np.ones_like(P2)
    
    # standard == True means quotas on rows; otherwise, quotas on columns.
    standard = (quota.shape[0] == m)
    quota = quota.flatten()
    
    # Replicate the side with quotas.
    if standard:    
        P1 = np.repeat(P1, quota, axis=0)
        P2 = np.repeat(P2, quota, axis=0)
    else:
        P1 = np.repeat(P1, quota, axis=1)
        P2 = np.repeat(P2, quota, axis=1)

    m, n = P1.shape  # New dimensions after replication

    # Always let columns propose. In both cases, we build P1_T from P1.
    P1_T = np.empty((m, n), dtype=int)
    for col in range(n):
        P1_T[:, col] = np.argsort(P1[:, col])
    # print("P1_T:", P1_T)

    NumStages = 0
    # For each (replicated) column, we store a pointer and a flag indicating if it should propose.
    cols_assigned = [[-1, True] for _ in range(n)]
    # For each row, we keep a list of columns that have proposed to it.
    rows_assigned = [[] for _ in range(m)]
    
    collisions = True
    while collisions:
        collisions = False
        NumStages += 1

        # Each column that needs to propose does so by using its pointer into its ordering in P1_T.
        for i in range(n):
            if cols_assigned[i][1]:
                cols_assigned[i][1] = False
                cols_assigned[i][0] += 1
                if cols_assigned[i][0] >= m: 
                    continue  # This column has exhausted its list.
                row = P1_T[cols_assigned[i][0]][i]
                rows_assigned[row].append(i)

        # Each row resolves collisions by keeping the best column (according to P2).
        for i in range(m):
            if len(rows_assigned[i]) <= 1:
                continue

            collisions = True
            # Choose the best proposal (the column with the lowest P2[i][col]).
            minRank, minCol = P2[i][rows_assigned[i][0]], rows_assigned[i][0]
            for col in rows_assigned[i]:
                if P2[i][col] < minRank:
                    minRank, minCol = P2[i][col], col
            
            # For every other column that proposed, mark it to propose again.
            for col in rows_assigned[i]:
                if col != minCol:
                    cols_assigned[col][1] = True
                    
            rows_assigned[i] = [minCol]
    
    # Build the match matrix from the final assignments.
    Match = np.zeros_like(P1_T)
    for row in range(m):
        for assigned in rows_assigned[row]:
            Match[row][assigned] = 1

    Match = combine_by_summing(Match, quota, axis=0) if standard else combine_by_summing(Match, quota, axis=1)
    return Match, NumStages

if __name__ == "__main__":
    questions = [loadmat('Q1.mat'), loadmat('Q2.mat'), loadmat('Q3.mat')]

    print("--------------- HOSPITAL QUESTIONS (2/2) ---------------\n")
    for question in questions:
        print("------------- NUMBER", questions.index(question)+1, "-------------")
        print(question.get("Quota").T if questions.index(question) != 1 else question.get("Quota"))

        print(question.get('P2'), "\n\n", question.get('P1'), "\n")

        match, numStages = GaleShapleyAlgorithmQuota(question.get('P2'), question.get('P1'), question.get("Quota").T if questions.index(question) != 1 else question.get("Quota"))
        print("Match Matrix:\n", match)
        print("Number of Stages:", numStages)
        match2, numStages2 = GaleShapleyAlgorithmQuota(question.get('P1').T, question.get('P2').T, question.get("Quota").T if questions.index(question) == 1 else question.get("Quota"))
        print("Match Matrix:\n", match2.T)
        print("Number of Stages:", numStages2)

