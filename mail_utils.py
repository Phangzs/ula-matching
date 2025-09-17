# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
# For email
import os, base64
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Additional email imports for attachments
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


# For markdown support
import markdown2
import json

# For email formatting
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape, TemplateNotFound
import random

# For interface
from tqdm import tqdm

# For reading files
import csv, re
import pandas as pd

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def creds():
    c=None
    if os.path.exists("token.json"):
        c = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not c or not c.valid:
        if c and c.expired and c.refresh_token:
            c.refresh(Request())
        else:
            c = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES).run_local_server(port=0)
        with open("token.json","w") as f: f.write(c.to_json())
    return c

# def send_email(to, subject, body):
#     service = build("gmail","v1", credentials=creds())
#     msg = MIMEText(body, "html", "utf-8")
#     msg["to"], msg["from"], msg["subject"] = to, "_", subject
#     raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()

def send_email(to, subject, body, attachments=None):
    service = build("gmail", "v1", credentials=creds())
    attachments = attachments or []

    if attachments:
        # Use multipart only when we have files
        msg = MIMEMultipart()
        msg["to"], msg["from"], msg["subject"] = to, "_", subject
        msg.attach(MIMEText(body, "html", "utf-8"))

        for path in attachments:
            ctype, encoding = mimetypes.guess_type(path)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)

            with open(path, "rb") as f:
                part = MIMEBase(maintype, subtype)
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            f'attachment; filename="{os.path.basename(path)}"')
            msg.attach(part)
    else:
        # No files → your original simple HTML email
        msg = MIMEText(body, "html", "utf-8")
        msg["to"], msg["from"], msg["subject"] = to, "_", subject

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()



def load_email_dict(path: str) -> dict:
    with open(f"{path}.json", 'r') as file:
        python_dict_from_file = json.load(file)
    return python_dict_from_file


env = Environment(
    loader=FileSystemLoader("templates"),
    undefined=StrictUndefined,
    autoescape=select_autoescape(["html", "xml"])
)

def render_email(template_name: str, *, custom_dict=None) -> tuple[str, str, str | None]:
    if not custom_dict: data = load_email_dict("templates/" + template_name)
    else: data = custom_dict
    
    # print("data:", data)
    with open(f"templates/{template_name}.subject.j2", 'r') as file:
        for line in file: print(line)
    
    subject = env.get_template(f"{template_name}.subject.j2", "templates").render(data)

    try:
        html = markdown2.markdown(env.get_template(f"{template_name}.body.html.j2", "templates").render(data), extras={'breaks': {'on_newline': True}})
    except TemplateNotFound:
        html = None
    return subject, html

def _email_confirmation(subject: str, html_body: str, template: str, *, test_email: str = None) -> bool:
    if test_email is None: 
        print("No test email will be sent.")
    else: 
        print(f"A test email will be sent to {test_email}.")

        customized_html_body, _, attachments = random.choice(fill_from_csv(html_body, template))
        send_email(test_email, subject, customized_html_body, attachments)
        return False
    
    # print("Please confirm the following are correct:", end="\n\n\n")
    print()

    email_tuples = list(pd.read_csv(f"templates/{template}.csv", usecols=[0, 1]).itertuples(index=False, name=None))
    print("----------NAME AND EMAILS----------")
    print(f"{"NAME":20}, EMAIL")
    for name, email in email_tuples:
        print(f"{name:20}, {email}")
    print()

    print("----------VARIABLES----------")
    email_vars = load_email_dict("templates/" + template)
    for key, value in email_vars.items():
        print(f"{key}: {value}")
    print()

    print("----------SUBJECT----------")
    print(subject)
    print()

    print("----------MESSAGE(HTML)----------")
    print(html_body)
    print()

    final_confirm = input("Please confirm that the above are all correct [Y/n]: ")

    return final_confirm.lower() == 'y'

    
# def _read_individual_data(csv_data: str) -> tuple[], List[tuple]:


# def _customize_body(body, custom_dict: dict[any, any]):
#     for key in dict:
#         body.replace(f"[[{key}]]", custom_dict[key])



# Matches {{KEY}} and captures KEY
# RX = re.compile(r"\[\[([A-Za-z0-9_]+)\]\]") # if the next doesn't work
RX = re.compile(r"\[\[([^]]+)\]\]") # Allows spaces/puntuation in brackets

from ast import literal_eval

def fill_from_csv(template: str, csv_path: str, *, missing="leave") -> list[str, str, list[str]]:
    """
    template.txt uses {{HeaderName}} placeholders.
    data.csv has headers matching those names.
    missing: "leave" -> keep {{KEY}}; "blank" -> replace with "" ; or give a string.
    """
    
    results = []
    with open(f"templates/{csv_path}.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for row in tqdm(reader):
            def repl(m):
                key = m.group(1)
                val = row.get(key)
                if val not in (None, ""):
                    return val
                if missing == "leave":
                    return m.group(0)           # keep [[KEY]]
                if missing == "blank":
                    return ""                    # remove it
                return str(missing)              # custom fallback text
            
            cell = (row.get("attachments") or "").strip()
            attachments_list: list[str]
            if cell.startswith("["):  # try JSON first, then Python literal
                try:
                    parsed = json.loads(cell)
                    if isinstance(parsed, (list, tuple)):
                        attachments_list = [str(p).strip() for p in parsed if str(p).strip()]
                    else:
                        attachments_list = []
                except Exception:
                    try:
                        parsed = literal_eval(cell)
                        attachments_list = [str(p).strip() for p in parsed if str(p).strip()] if isinstance(parsed, (list, tuple)) else []
                    except Exception:
                        attachments_list = []
            else:
                # fallback: split on common delimiters
                attachments_list = [p.strip() for p in re.split(r"[;,|]", cell) if p.strip()]

            # attachments_list = [p.strip() for p in re.split(r"[;,|]", (row.get("attachments") or "")) if p.strip()]
            results.append((RX.sub(repl, template), row['email'].strip(), attachments_list))

    return results


def send_emails(template: str) -> None:
    # assert len(template_tuple) > 0, "Email tuple must not be empty!"


    subject, html_body = render_email(template)

    email_tuples = fill_from_csv(html_body, template)
    # print(email_tuples)

    test_email = input("If you would like to send a test email, enter your email. Otherwise, press return:\n")
    test_email = test_email if "@" in test_email else None
    if _email_confirmation(subject, html_body, template, test_email=test_email): print("Confirmation Complete: Sending emails...")
    else: 
        print("Stopping. No mass emails sent.")
        return

    


    for body, email, attachments in tqdm(email_tuples):
        send_email(email, subject, body, attachments)

    print("\nAll emails sent.")