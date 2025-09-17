from galeShapleyAlgorithmQuota import *
from tqdm import tqdm 
import random

def diffusedMatch(P1, P2, quotas):

    P1, P2 = matchMatrices(P1, P2)

    experience = P1[:, -1]
    P1 = P1[:, :-1]

    P1_match = P1[:, 1:].astype(int)
    P2_match = P2[:, 1:].astype(int)


    match1, _ = GaleShapleyAlgorithmQuota(P2_match, P1_match, quota=quotas, info=experience)


    p = np.random.permutation(P1.shape[0])


    incorrect_sum = 0
    incorrect = 0
    iterations = 10000
    total_match = np.zeros_like(P1)[:,1:]

    for i in tqdm(range(iterations)):
        p = np.random.permutation(P1.shape[0])
        s = np.empty(p.size, dtype=np.int32)
        for i in np.arange(p.size):
            s[p[i]] = i


        P1_rand, P2_rand = matchMatrices(P1[p], P2)

        match2, _ = GaleShapleyAlgorithmQuota(P2_match, P1_match, quota=quotas, info=experience)

        match2 = match2[s]

        total_match = total_match + match2

    diffused_matches = np.insert(total_match / iterations * 1000, 0, P1[:, 0], axis=1)


    diff = match1 * iterations - total_match
    incorrect = np.sum(np.abs(diff))



    print("\navg incorrect:", incorrect)
    print("avg avg total diff:", incorrect *1.0 / iterations)
    print("avg avg diff:", incorrect *1.0 / iterations / P1.shape[0])
    print("\n")


    maxMatch = greedyMaxProbMatch(diffused_matches, quotas)

    return maxMatch


def matchToDict(match_matrix, df_filtered) -> dict:
    # Don't want ID
    df_filtered_columns_list = df_filtered.columns.to_list()[1:]
    counter = 0

    match_dict = {}


    for i in range(len(match_matrix)):
        # Skip non-matched
        if np.all(match_matrix[i, 1:] == 0): continue

        counter += 1

        id = match_matrix[i,0]
        course_number = df_filtered_columns_list[np.nonzero(match_matrix[i, 1:])[0][0]][20:-1].split()[0]
        match_dict[id] = course_number

    return match_dict



def printMatches(match_dict: dict) -> None:
    print("ALL PAIRED MATCHES:")
    for id in match_dict: print(id, match_dict[id])
    print("\n")

# match_comparison = []
# correct = 0
# for reference in reference_dict:
#     aligned_match = reference, reference_dict.get(reference), match_dict.get(reference) if match_dict.get(reference) != None else "-"
#     match_comparison.append(aligned_match)
#     print(type(aligned_match[1]))
#     if aligned_match[1].lower().strip() == aligned_match[2].lower().strip(): 
#         correct += 1
#     else:
#         continue


def printExperienceData(match_dict: dict, df_original) -> None:

    # course_reference_inv = {}
    course_match_inv = {}
    # course_exp_reference = {}
    course_exp_match = {}
    # for course in set(reference_dict.values()): course_exp_reference[course] = [0, 1]
    for course in set(match_dict.values()): course_exp_match[course] = [0, 1]
    # for key, value in reference_dict.items():
    #     course_reference_inv.setdefault(value, []).append(key)
    for key, value in match_dict.items():
        course_match_inv.setdefault(value, []).append(key)

    # print(course_reference_inv)
    # print(course_match_inv)
    # print(course_exp_reference)
    # print(course_exp_match)

    # print("MATCHING PERFORMANCE RELATIVE TO GIVEN")
    # print(f"Identical Matches: {correct}, Total Matches: {len(match_comparison)}")
    # print(f"Relative Identical Matches: {round(correct/len(match_comparison)*100,2)}%\n")

    experience_dict = dict(zip(df_original["ID"], df_original["Previous LA Service for CS"]))

    # Build dictionary of ULAs per class
    classes_dict = {}

    for id in match_dict:
        if match_dict[id] not in classes_dict: classes_dict[match_dict[id]] = [id]
        else: classes_dict[match_dict[id]].append(id)

    # Build dictionary of whether or not each class has a mixed-ULA experience (at least 1 new, 1 experienced)
    class_experience_dict = classes_dict.copy()
    avoided_ids = {}
    for class_id in class_experience_dict:
        experienced, inexperienced = False, False
        for student_id in class_experience_dict[class_id]:
            student_id = student_id
            if student_id not in experience_dict:
                try: avoided_ids[student_id] = experience_dict[student_id]
                except: avoided_ids[student_id] = experience_dict[str(student_id)]
                continue
            experienced = experienced or experience_dict[student_id] > 0
            inexperienced = inexperienced or experience_dict[student_id] <= 0

        if experienced and inexperienced: class_experience_dict[class_id] = True
        elif experienced: class_experience_dict[class_id] = "No inexperienced"
        elif inexperienced: class_experience_dict[class_id] = "No experienced"
        else: class_experience_dict[class_id] = False
        
    if len(avoided_ids) > 0: print("Skipped matches:", avoided_ids)
    else: print("No skipped matches.")
    print(f"Proportion of mixed exp classes: {round(list(class_experience_dict.values()).count(True) / len(class_experience_dict)*100, 2)}%")
    
def match(P1: pd.DataFrame, P2: pd.DataFrame, quotas: list[list], df_original: pd.DataFrame, df_filtered: pd.DataFrame, *, print_statistics=False):
    match_dict = diffusedMatch(P1, P2, quotas) 

    match_dict = matchToDict(match_dict, df_filtered)

    if print_statistics: 
        printMatches(match_dict)
        printExperienceData(match_dict, df_original)

    return match_dict

