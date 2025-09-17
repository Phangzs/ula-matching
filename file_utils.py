from mail_utils import send_emails
import pandas as pd
import numpy as np
import regex as re

def load_csv(filename):
    # Load the CSV file (first row is header)
    df = pd.read_csv(filename)

    # Get column names
    all_columns = df.columns.tolist()


    # col[len("Course Preferences")+2:-1]
    matching_columns = [col for col in df.columns if re.search("Course Preferences", col)] # This assumes that there is a closing bracket after the course
    if matching_columns == None: exit(1)

    choice_columns = matching_columns



    # Now filter the DataFrame to only include the desired columns:
    # "ID", the choice columns, "First Choice", and "Second Choice"
    cols_to_keep = ["ID"] + choice_columns + ["First Choice", "Second Choice"]
    df_filtered = df[cols_to_keep]

    # Prepare a new matrix: one row per ID and one column per choice column (G–T)
    # Initialize with zeros (assuming that if neither choice matches, the value stays 0)
    matrix = np.empty((df_filtered.shape[0], len(choice_columns)), dtype=object)

    # matrix = np.zeros((df_filtered.shape[0], len(choice_columns)), dtype=str)

    # Pre-populate the matrix with (first character as integer + 2) for each cell in columns G-T
    for i, row in df_filtered.iterrows():
        for j, col in enumerate(choice_columns):
            try:
                matrix[i, j] = int(str(row[col]).strip()[0]) + 2
            except (ValueError, IndexError):
                pass
    
    # Make sure normalized columns contain only the normalized text for matching
    normalized_cols = [c[20:-1].strip().lower() for c in choice_columns]
    normalized_cols = [c for c in normalized_cols if c]
    # print(normalized_cols)

    # Process each row of the filtered DataFrame.
    # For each row, find the index (in choice_columns) for the first and second choices
    # and assign 1 or 2 accordingly.
    for idx, row in df_filtered.iterrows():
        if not pd.isnull(row["First Choice"]): first_choice = row["First Choice"].strip().lower()
        if not pd.isnull(row["Second Choice"]): second_choice = row["Second Choice"].strip().lower()
        
        # Set value 1 for first choice if the header exists in choice_columns
        if first_choice in normalized_cols:
            col_index = normalized_cols.index(first_choice)
            matrix[idx, col_index] = 1
        
        # Set value 2 for second choice if the header exists in choice_columns
        if second_choice in normalized_cols:
            col_index = normalized_cols.index(second_choice)
            matrix[idx, col_index] = 2

    # print(df_filtered[df_filtered.columns[0]])
    ids = np.array(df_filtered[df_filtered.columns[0]])
    print("ids:", ids)
    print("matrix before:", matrix)
    # print(ids, ids.shape)
    # print("Matrix before:", matrix)
    # print(matrix.shape)
    # np.column_stack((ids, matrix))
    matrix = np.insert(matrix, 0, ids, axis=1)
    print("matrix after:", matrix)
    matrix = np.insert(matrix, matrix.shape[1], df["Previous LA Service for CS"].to_numpy(), axis=1)
    # matrix = np.append(ids, matrix).reshape((matrix.shape[0], matrix.shape[1]+1))
    # np.concatenate(ids, matrix)
    # np.insert(matrix, range(matrix.shape[0]), ids)
    # print("matrix: ", matrix, matrix.shape)
    # for row in range(len(matrix)):


    # # dataframe if needed
    # result_df = pd.DataFrame(matrix, columns=choice_columns)
    return matrix, df_filtered, df

def load_prof(filename):
    df = pd.read_csv(filename, index_col=False).fillna(99)


    # Convert from floats to ints
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype(int)
    

    df_formatted = df
    
    # Get only desired columns (in-place)
    df_formatted = df_formatted.iloc[:, np.r_[0, 2:df_formatted.shape[1]]] 

    df_formatted = df_formatted[1:]
    df_formatted = np.array(df_formatted, dtype=str)

    return df_formatted, df

def verifyEmail(prof_name, dictionary):
    try:
        email = dictionary[prof_name]
    except:
        print(f"No email for {prof_name}!")
        return prof_name, None
    return prof_name, email
import os

def courseTopChoices(df) -> list:
    # 1. Find First and Second Choice columns (should be consecutive)
    # 2. For each possible course (find start and end of course name columns)
    # 2.5. Create email dictionary
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


    # Part 2.5 
    data = pd.read_csv("email_directory.csv")
    data = dict(zip(data['Professor'], data['email']))


    # Part 3 & 4

    # First make sure professor emails exist in the directory
    matched_emails = []
    unmatched_names = []
    for course in matching_columns:
        prof = course[course.find("(")+1:course.find(")")]

        # print(f"Professor: {prof}")
        
        name, email = verifyEmail(prof, data)

        if email == None: unmatched_names.append(name)
        else: matched_emails.append((name, email))

    if len(unmatched_names) != 0:
        print("WARNING: Cannot find emails associated with some courses. A temporary email will be requested.")
        # print("\nPlease update email_directory.csv with all the relevant emails and try again.")
        # print("No emails sent.")
        # return

    # Then generate the spreadsheets
    email_tuples = [["professor", "email", "attachments", "quota"]]
    all_csvs = []
    all_dfs = []
    for course in matching_columns:
        prof = course[course.find("(")+1:course.find(")")]
        course_df = df[(df['First Choice'] == course) | (df['Second Choice'] == course)]
        course_df.insert(0, 'Ranking', '')

        name, email = verifyEmail(prof, data)

        if email == None: 
            email = input(f"Email not found for {name}. Please enter associated email (the directory will not be updated with this name and email). Enter nothing to skip emailing this course:\n")
            if email == "":
                print(f"{course} will be skipped")
                continue


        csv_path = course+'.csv'

        course_df.to_csv(csv_path, index=False)
        all_csvs.append(csv_path)
        all_dfs.append((csv_path, course_df))

        quota = input(f"Please enter the quota for {course}:\n")

        email_tuples.append([name, email, [csv_path], quota])

        if quota == "":
            print("No quota detected. Abandoning quota assignments. Please try again.")
            break

        # print(course)

        # TODO: Email CSV File

        # # Import smtplib for the actual sending function
        # import smtplib

        # # Import the email modules we'll need
        # from email.message import EmailMessage

        # # Open the plain text file whose name is in textfile for reading.
        # textfile = "test.txt"
        # with open(textfile) as fp:
        #     # Create a text/plain message
        #     msg = EmailMessage()
        #     msg.set_content(fp.read())

        # # me == the sender's email address
        # # you == the recipient's email address
        # msg['Subject'] = f'The contents of {textfile}'
        # msg['From'] = "erikfeng@ucsb.edu"
        # msg['To'] = "erikfeng@ucsb.edu"

        # # Send the message via our own SMTP server.
        # s = smtplib.SMTP('localhost')
        # s.send_message(msg)
        # s.quit()


    
    df = pd.DataFrame(email_tuples[1:], columns=email_tuples[0])
    # df = pd.DataFrame(email_tuples)
    df.to_csv("templates/professor_preferences.csv", index=False)
    if quota != "": send_emails("professor_preferences")

    for csv_file in all_csvs:
        os.remove(csv_file)

    return all_dfs


from pathlib import Path

def recombine_preferences(folder: str = "completed_preferences", out_csv: str = "prof_preferences.csv"):
    folder = Path(folder)
    csv_files = sorted(folder.glob("*.csv"))

    series = []
    for fp in csv_files:
        name = fp.stem  # spreadsheet name without .csv
        df = pd.read_csv(fp, usecols=["Ranking", "ID", "Previous LA Service for CS"])
        s = (
            df.drop_duplicates(subset=["ID", "Previous LA Service for CS"], keep="last")
              .set_index(["ID", "Previous LA Service for CS"])["Ranking"]
              .rename(name)
        )
        series.append(s)

    combined = pd.concat(series, axis=1).reset_index()
    combined = combined[["ID", "Previous LA Service for CS"] + [s.name for s in series]]
    combined.to_csv(out_csv, index=False)

def getQuotas(file: str = "professor_preferences.csv") -> list[list]:
    quotas = pd.read_csv(f"templates/{file}", usecols=["quota"]).to_numpy()
    return quotas

def createAcceptanceTracker(form_submissions: str, match_dict: dict, *, tracker_path="student_responses.csv") -> str:
    students = pd.read_csv(form_submissions, usecols=["ID", "First name", "Last name", "Email Address"], dtype=str) # Expand this to first and last name for more use

    df = students.assign(
        **{
            "assignment": pd.Series("", index=students.index),
            "status": "unconfirmed"
        }
    )

    assignments = df["ID"].map(match_dict)
    mask = assignments.notna()

    df.loc[mask, "assignment"] = assignments[mask]
    df.loc[mask, "status"] = "pre-notified"

    # send_emails()

    df.to_csv(tracker_path, index=False)

    return tracker_path

def sendRejectionEmails(tracker_path: str = "student_responses.csv") -> None:
    assignments = pd.read_csv(tracker_path, dtype=str)
    rejected_students = assignments[assignments["assignment"].isnull()]
    rejected_students = rejected_students[["Email Address", "First name"]]
    rejected_students.columns = ["email", "applicant"]
    rejected_students = rejected_students.iloc[:, ::-1] # Reverse the order

    rejected_students.to_csv("templates/ula_rejection.csv", index=False)

    send_emails("ula_rejection")

