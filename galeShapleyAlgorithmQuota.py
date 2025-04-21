import numpy as np
import pandas as pd
from scipy.io import loadmat
import regex as re
import math

def combine_by_summing(arr, groups, axis=0):
    groups = np.array(groups, dtype=int)
    
    # Check that the groups sum to the size along the given axis.
    if groups.sum() != arr.shape[axis]: exit(1)
    
    # Compute the starting indices of each group.
    indices = np.concatenate(([0], np.cumsum(groups)[:-1]))
    
    # Use np.add.reduceat to sum over the specified groups.
    return np.add.reduceat(arr, indices, axis=axis)

# TODO: Check randomness

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

def load_csv(filename):
    # Load the CSV file (first row is header)
    df = pd.read_csv(filename)

    # Get column names
    all_columns = df.columns.tolist()

    # column G (7th column) = index 6 (0-indexed),
    # column T (20th column) = index 19.
    choice_columns = all_columns[6:20]

    # Now filter the DataFrame to only include the desired columns:
    # "ID", the choice columns, "First Choice", and "Second Choice"
    cols_to_keep = ["ID"] + choice_columns + ["First Choice", "Second Choice"]
    df_filtered = df[cols_to_keep]

    # Prepare a new matrix: one row per ID and one column per choice column (G–T)
    # Initialize with zeros (assuming that if neither choice matches, the value stays 0)
    matrix = np.zeros((df_filtered.shape[0], len(choice_columns)), dtype=int)

    # Pre-populate the matrix with (first character as integer + 2) for each cell in columns G-T
    for i, row in df_filtered.iterrows():
        for j, col in enumerate(choice_columns):
            try:
                matrix[i, j] = int(str(row[col]).strip()[0]) + 2
            except (ValueError, IndexError):
                pass

    
    # Make sure normalized columns contain only the normalized text for matching
    normalized_cols = [c[20:-1].strip().lower() for c in choice_columns]
    # print(normalized_cols)

    # Process each row of the filtered DataFrame.
    # For each row, find the index (in choice_columns) for the first and second choices
    # and assign 1 or 2 accordingly.
    for idx, row in df_filtered.iterrows():
        # print(row["First Choice"], row["Second Choice"], "second choice nan:", pd.isnull(row["Second Choice"]))
        # print(bool(first_choice), bool(second_choice))
        if not pd.isnull(row["First Choice"]): first_choice = row["First Choice"].strip().lower()
        if not pd.isnull(row["Second Choice"]): second_choice = row["Second Choice"].strip().lower()
        # print(first_choice, second_choice)
        
        # Set value 1 for first choice if the header exists in choice_columns
        # print(first_choice, first_choice in normalized_cols)
        if first_choice in normalized_cols:
            col_index = normalized_cols.index(first_choice)
            # print(col_index, 1)
            matrix[idx, col_index] = 1
        
        # Set value 2 for second choice if the header exists in choice_columns
        if second_choice in normalized_cols:
            col_index = normalized_cols.index(second_choice)
            matrix[idx, col_index] = 2

    # # dataframe if needed
    # result_df = pd.DataFrame(matrix, columns=choice_columns)
    return matrix, df_filtered, df

def courseTopChoices(df, export_path=""):
    # 1. Find First and Second Choice columns (should be consecutive)
    # 2. For each possible course (find start and end of course name columns)
    # 3. Filter by those values per course, create a spreadsheet
    # 4. Return spreadsheets (OR PRODUCE CSV FILES?)

    # First Choice name
    first_choice_column_name = "First Choice"
    course_string = "Course Preferences"

    # Part 1

    first_choice_index = df.columns.get_loc(first_choice_column_name)

    # Part 2

    # Find column of all occurance of courses
    matching_columns = [col[len(course_string)+2:-1] for col in df.columns if re.search(course_string, col)] # This assumes that there is a closing bracket after the course
    if matching_columns == None: exit(1)

    # Part 3

    for course in matching_columns:

        course_df = df[(df['First Choice'] == course) | (df['Second Choice'] == course)]

        course_df.to_csv(export_path+course+'.csv', index=False)
        
        # print("course:", course)

    

    

    # df[df['First Choice']]

if __name__ == "__main__":
    
    with np.printoptions(threshold=np.inf):
        print(load_csv("Example_ULA_Applications.csv")[0])
    

    # exit(0)

    P1, df_filtered, df_original = load_csv("Example_ULA_Applications.csv")
    # print(P1.shape[0])
    test_quotas = np.ones((P1.shape[0],1)).astype(int) * 3
    test_quotas = np.ones((1,P1.shape[1])).astype(int) * 3
    print(test_quotas)

    # courseTopChoices(df_original)

    # exit(0)


    p = np.random.permutation(P1.shape[0])



    match1, _ = GaleShapleyAlgorithmQuota(P1, np.ones_like(P1), quota=test_quotas)

    p = np.random.permutation(P1.shape[0])
    s = np.empty(p.size, dtype=np.int32)
    for i in np.arange(p.size):
        s[p[i]] = i
    match2, _ = GaleShapleyAlgorithmQuota(P1[p], np.ones_like(P1), quota=test_quotas)

    with np.printoptions(threshold=np.inf):
        print(P1-(P1[p])[s])
        print(match1-match2)


    
    exit(0)

    with np.printoptions(threshold=np.inf):
        print(match)

    print(np.count_nonzero(match))

    print(df_filtered)
    # print(np.nonzero(match[0])[0][0])

    # Don't want ID
    df_filtered_columns_list = df_filtered.columns.to_list()[1:]

    # print("ID:", df_filtered.iloc[51]["ID"], "Course:", df_filtered_columns_list[np.nonzero(match[51])[0][0]][20:-1])
    # print(df_filtered_columns_list)
    print("MATCHES (ID, COURSE):")
    for i in range(len(match)):
        # Skip non-matched
        if np.all(match[i] == 0): continue

        print(df_filtered.iloc[i]["ID"], df_filtered_columns_list[np.nonzero(match[i])[0][0]][20:-1])

    

