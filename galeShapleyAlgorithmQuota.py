import numpy as np
import pandas as pd
import regex as re

from natsort import natsorted
from mail_utils import send_emails

def matchMatrices(P1, P2):

    ### FILTER MATRICES TO COMMON IDS
    # 1) find the common IDs
    common_ids = np.intersect1d(P1[:,0], P2[:,0])
    # common_ids == array([10, 20])

    # 2) build boolean masks for each matrix
    maskA = np.isin(P1[:,0], common_ids)
    maskB = np.isin(P2[:,0], common_ids)

    # 3) filter
    P1 = P1[maskA]
    P2 = P2[maskB]

    ### SORT MATRICES
    # 1) pull out the ID columns
    ids1 = P1[:, 0]
    ids2 = P2[:, 0]

    # print(ids1, ids2)

    # 2) build a lookup from ID → row‐index in mat2
    pos_in_P2 = { id_: i for i, id_ in enumerate(ids2) }

    # 3) for each id in mat1, find its row in mat2
    order = [pos_in_P2[id_] for id_ in ids1]

    # 4) fancy‐index to reorder mat2
    P2_matched = P2[order]

    # print(P2_matched)
    return P1, P2_matched


def combine_by_summing(arr, groups, axis=0):
    groups = np.array(groups, dtype=int)
    
    # Check that the groups sum to the size along the given axis.
    if groups.sum() != arr.shape[axis]: exit(1)
    
    # Compute the starting indices of each group.
    indices = np.concatenate(([0], np.cumsum(groups)[:-1]))
    
    # Use np.add.reduceat to sum over the specified groups.
    return np.add.reduceat(arr, indices, axis=axis)

# TODO: Check randomness
def GaleShapleyAlgorithmQuota(P1, P2, quota, info=None):
    m, n = P1.shape  # m = size of Group 2, n = size of Group 1
    orig_m, orig_n = m, n  

    # Convert from 1-indexed to 0-indexed
    P1 = P1 - np.ones_like(P1)
    P2 = P2 - np.ones_like(P2)
    
    # standard == True means quotas on rows; otherwise, quotas on columns.
    standard = (quota.shape[0] == m)
    quota = quota.flatten()  # keep as vector of replication counts
    
    # We will fill `info_rep` after replication; `col2orig` after we know n.
    info_rep = None  # ADDED
    # ---- replicate the side with quotas (unchanged logic) ----
    if standard:    
        P1 = np.repeat(P1, quota, axis=0)
        P2 = np.repeat(P2, quota, axis=0)
    else:
        P1 = np.repeat(P1, quota, axis=1)
        P2 = np.repeat(P2, quota, axis=1)

    m, n = P1.shape  # New dimensions after replication

    # replicate info if provided
    if info is not None:
        info = np.asarray(info, dtype=int).flatten()
        if info.shape[0] != orig_m:
            raise ValueError("`info` must have length equal to the number of rows before replication.")
        info_rep = np.repeat(info, quota, axis=0) if standard else info.copy()

    # map each (possibly replicated) column index -> original column index
    if standard:
        # columns were not replicated
        col2orig = np.arange(n, dtype=int)
    else:
        # columns were replicated; expand mapping using `quota` over original columns
        col2orig = np.concatenate([np.full(q, j, dtype=int) for j, q in enumerate(quota)])

    # Always let columns propose. In both cases, we build P1_T from P1.
    P1_T = np.empty((m, n), dtype=int)
    for col in range(n):
        P1_T[:, col] = np.argsort(P1[:, col])

    NumStages = 0
    # For each (replicated) column, we store a pointer and a flag indicating if it should propose.
    cols_assigned = [[-1, True] for _ in range(n)]
    # For each row, we keep a list of columns that have proposed to it.
    rows_assigned = [[] for _ in range(m)]
    
    collisions = True
    while collisions:
        collisions = False
        NumStages += 1

        # has_match[c]  : whether this original column currently holds >=1 match
        # all_pos[c]    : all its matched rows have info>0
        # all_zero[c]   : all its matched rows have info==0
        if info_rep is not None:
            num_orig_cols = col2orig.max() + 1
            has_match = np.zeros(num_orig_cols, dtype=bool)
            all_pos   = np.ones(num_orig_cols,  dtype=bool)
            all_zero  = np.ones(num_orig_cols,  dtype=bool)
            for r in range(m):
                if rows_assigned[r]:
                    j = rows_assigned[r][0]            # the (replicated) column currently holding row r
                    oc = col2orig[j]                   # original column index
                    has_match[oc] = True
                    v = int(info_rep[r])
                    if v > 0:
                        all_zero[oc] = False
                    elif v == 0:
                        all_pos[oc] = False
                    else:  # negative -> treat as ">0"
                        all_zero[oc] = False
        else:
            has_match = all_pos = all_zero = None  # no info => no special tie-breaking

        # Each column that needs to propose does so by using its pointer into its ordering in P1_T.
        for i in range(n):
            if cols_assigned[i][1]:
                cols_assigned[i][1] = False
                cols_assigned[i][0] += 1
                p = cols_assigned[i][0]
                if p >= m:
                    continue  # This column has exhausted its list.

                # tie-break among equal-best rows at this preference level using `info`
                # Only if the column already has >=1 match and we have `info`.
                if info_rep is not None:
                    oc = col2orig[i]
                    if has_match is not None and has_match[oc]:
                        # Current best preference value for this column at pointer p
                        cand_row = P1_T[p, i]
                        best_val = P1[cand_row, i]

                        # Scan the tied block [p .. end of that value]
                        best_q = p
                        if all_pos[oc]:
                            # All current matches have info>0 → prefer a row with info==0 if available
                            q = p
                            while q < m and P1[P1_T[q, i], i] == best_val:
                                r = P1_T[q, i]
                                if info_rep[r] == 0:
                                    best_q = q
                                    break
                                q += 1
                        elif all_zero[oc]:
                            # All current matches have info==0 → prefer a row with info>0 if available
                            q = p
                            while q < m and P1[P1_T[q, i], i] == best_val:
                                r = P1_T[q, i]
                                if info_rep[r] > 0:
                                    best_q = q
                                    break
                                q += 1
                        # Mixed (both 0 and >0 present) → keep original order

                        # If we found a better q inside the tied block, swap so pointer p picks it now
                        if best_q != p:
                            P1_T[p, i], P1_T[best_q, i] = P1_T[best_q, i], P1_T[p, i]

                row = P1_T[p, i]
                rows_assigned[row].append(i)

        # Each row resolves collisions by keeping the best column (according to P2).
        for i in range(m):
            if len(rows_assigned[i]) <= 1:
                continue

            collisions = True
            # Choose the best proposal (the column with the lowest P2[i][col]), and record ties
            minRank, minCol, ties = P2[i][rows_assigned[i][0]], rows_assigned[i][0], 0
        
            for col in rows_assigned[i]:
                if P2[i][col] < minRank:
                    minRank, minCol, ties = P2[i][col], col, 0
                elif P2[i][col] == minRank:
                    ties += 1
            
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



def greedyMaxProbMatch(dmatch, quotas):
    final_match = np.zeros_like(dmatch[:, 1:])

    dmatch_temp = dmatch.copy()[:, 1:]

    temp_quotas = quotas.copy()

    while sum(temp_quotas) > 0:
        
        max_index = np.argmax(dmatch_temp)

        row_index, col_index = np.unravel_index(max_index, final_match.shape)

        # print(max_index, dmatch_temp)

        dmatch_temp[row_index, col_index] = 0

        # print(np.sum(final_match[row_index]), temp_quotas[col_index][0])
        if np.sum(final_match[row_index]) > 0 or temp_quotas[col_index][0] <= 0: continue

        final_match[row_index, col_index] = 1

        
        temp_quotas[col_index][0] -= 1
        # print(temp_quotas, temp_quotas[col_index])
    
    return np.insert(final_match, 0, dmatch[:, 0], axis=1)
