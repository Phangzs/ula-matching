import numpy as np
from scipy.io import loadmat


def GaleShapleyAlgorithm(P1, P2):
    m, n = P1.shape  # m = size of Group 2, n = size of Group 1

    # Convert rank to index
    P1 = P1 - np.ones_like(P1)
    P2 = P2 - np.ones_like(P2)

    # Step 1: reformat P1 into P1_T (choices in order)

    P1_T = np.zeros((m,n), dtype=int)
    for row in range(m):
        for col in range(n):
            P1_T[P1[row][col]][col] = row

    # Step 2: Create assigned lists
    NumStages = 0
    cols_assigned = [[-1, True] for _ in range(n)] # (index, shift to next?)
    rows_assigned = [[] for _ in range(m)]
    

    # Step 3: Select choices:
    collisions = True
    while collisions:
        collisions = False
        NumStages += 1

        for i in range(n):
            if cols_assigned[i][1]:
                # Update col
                cols_assigned[i][1] = False
                cols_assigned[i][0] += 1

                # Sanity Check
                if cols_assigned[i][0] >= m: continue

                # Update row
                rows_assigned[P1_T[cols_assigned[i][0]][i]].append(i)

        for i in range(m):
            if len(rows_assigned[i]) <= 1: continue

            collisions = True
            
            # Remove non-best match from row assignment
            # Mark columns for matching next round
            minRank, minCol = P2[i][rows_assigned[i][0]], rows_assigned[i][0]
            for col in rows_assigned[i]: 
                if P2[i][col] < minRank:
                    minRank, minCol = P2[i][col], col
            
            for col in rows_assigned[i]:
                if col == minCol: continue
                cols_assigned[col][1] = True
                
            rows_assigned[i] = [minCol]
    
    Match = np.zeros_like(P1)
    for row in range(m):
        for assigned in rows_assigned[row]:
            Match[row][assigned] = 1

    return Match, NumStages

# print(GaleShapleyAlgorithm(np.array([[1,1,2,3,3],[2,3,1,1,2],[3,2,3,2,1]]),np.array([[2,1,3,4,5],[3,1,2,5,4],[3,1,4,2,5]])))
# exit(0)

if __name__ == "__main__":

    # QUESTION 1
    print("-------------QUESTION 1-------------")
    P1 = np.array([[3,3,2,3],
                   [4,1,3,2],
                   [2,4,4,1],
                   [1,2,1,4]])
    P2 = np.array([[1,2,3,4],
                   [1,4,3,2],
                   [2,1,3,4],
                   [4,2,3,1]])

    match, numStages = GaleShapleyAlgorithm(P1, P2)
    print("Input matrices:\nP1:\n", P1, "\nP2:\n", P2, "\n")
    print("Match Matrix (Riders proposing):\n", match)
    print("Number of Stages:", numStages, end="\n\n")
    match, numStages = GaleShapleyAlgorithm(P2.T, P1.T)
    print("Match Matrix (Drivers proposing):\n", match.T)
    print("Number of Stages:", numStages, end="\n\n")

    # QUESTION 2
    print("-------------QUESTION 2-------------")
    P1 = np.array([[1,1,2,3,3],
                   [2,3,1,1,2],
                   [3,2,3,2,1]])
    P2 = np.array([[2,1,3,4,5],
                   [3,1,2,5,4],
                   [3,1,4,2,5]])

    match, numStages = GaleShapleyAlgorithm(P1, P2)
    print("Input matrices:\nP1:\n", P1, "\nP2:\n", P2, "\n")
    print("Match Matrix (Riders proposing):\n", match)
    print("Number of Stages:", numStages, end="\n\n")
    match, numStages = GaleShapleyAlgorithm(P2.T, P1.T)
    print("Match Matrix (Drivers proposing):\n", match.T)
    print("Number of Stages:", numStages, end="\n\n")




    questions = [loadmat('Q1.mat'), loadmat('Q2.mat'), loadmat('Q3.mat')]
    print("--------------- HOSPITAL QUESTIONS (1/2) ---------------\n")
    for question in questions:
        print("NUMBER", questions.index(question)+1)
        # print(question.get("Quota"))

        print(question.get('P2'), "\n\n", question.get('P1'), "\n")

        match, numStages = GaleShapleyAlgorithm(question.get('P2'), question.get('P1'))
        print("Match Matrix:\n", match)
        print("Number of Stages:", numStages)

        match2, numStages2 = GaleShapleyAlgorithm(question.get('P1').T, question.get('P2').T)
        print("Match Matrix:\n", match2.T)
        print("Number of Stages:", numStages2, end="\n\n\n")