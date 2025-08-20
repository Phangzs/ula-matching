# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
# For email
import os, base64
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

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

def send_email(to, subject, body):
    service = build("gmail","v1", credentials=creds())
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

def _email_confirmation(email_tuples: list[tuple[str, str]], subject: str, html_body: str, template: str, *, test_email: str = None, name_variable: str = "name") -> bool:
    if test_email is None: 
        print("No test email will be sent.")
    else: 
        rand_name = random.choice(email_tuples)[0]
        print(f"A test email will be sent to {test_email} for confirmation a random name ({rand_name}).")

        customized_html_body = html_body.replace("[["+name_variable+"]]", rand_name)
        send_email(test_email, subject, customized_html_body)
        return False
    
    # print("Please confirm the following are correct:", end="\n\n\n")
    print()

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

    





def send_emails(email_tuples: list[tuple[str, str]], template: str, *, name_variable="name") -> None:
    assert len(email_tuples) > 0, "Email tuple must not be empty!"


    subject, html_body = render_email(template)

    test_email = input("If you would like to send a test email, enter your email. Otherwise, press return:\n")
    test_email = test_email if "@" in test_email else None
    if _email_confirmation(email_tuples, subject, html_body, template, test_email=test_email, name_variable=name_variable): print("Confirmation Complete: Sending emails...")
    else: 
        print("Stopping. No mass emails sent.")
        return

    for name, email in tqdm(email_tuples):
        customized_html_body = html_body.replace("[["+name_variable+"]]", name)
        send_email(email, subject, customized_html_body)

    print("All emails sent.")