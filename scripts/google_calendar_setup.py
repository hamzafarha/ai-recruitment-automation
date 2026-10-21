"""Authorise the app once to create interview events with a Google Meet link in your Google Calendar.

    python scripts/google_calendar_setup.py

Before (once, about 5 minutes, docs/USER_GUIDE.md section "Google Meet invitations"):
1. https://console.cloud.google.com -> create a project -> "APIs & Services" -> enable "Google Calendar API";
2. "OAuth consent screen": External, add your Gmail address as a test user;
3. "Credentials" -> "Create credentials" -> "OAuth client ID" -> Desktop app -> download the JSON as
   secrets/google_client_secret.json (GOOGLE_CLIENT_SECRETS in .env).
This script opens the browser, you sign in and accept, and it saves the token to secrets/google_token.json
(GOOGLE_TOKEN_FILE). Then MEETING_PROVIDER=auto (or google) creates a Google Calendar event with a Meet
link for every interview a recruiter schedules, and Google e-mails the invitation to the candidate.
The secrets/ folder is ignored by git: never commit it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        sys.exit('The Google libraries are not installed: pip install -e ".[google]"')
    from recruitment_ai.config.settings import get_settings
    from recruitment_ai.services.meetings import GOOGLE_SCOPES, project_path

    settings = get_settings()
    secrets = project_path(settings.google_client_secrets)
    token = project_path(settings.google_token_file)
    if not secrets.is_file():
        sys.exit(f"No OAuth client file at {secrets}.\nCreate a Desktop OAuth client in Google Cloud (see this "
                 "script's header) and save its JSON there.")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), GOOGLE_SCOPES)
    credentials = flow.run_local_server(port=0, prompt="consent")
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text(credentials.to_json(), encoding="utf-8")
    print(f"Saved {token}. Interviews now get a Google Meet link (MEETING_PROVIDER=auto or google); restart the API.")


if __name__ == "__main__":
    main()
