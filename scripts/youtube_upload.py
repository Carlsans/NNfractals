#!/usr/bin/env python3
"""Uploads a video to YouTube for the queue manager (nnfractals-queue), with the
YouTube Data API v3 (resumable upload, retried on network and server errors).

Setup, once:
  - the uploader's own Python: ~/.local/share/nnfractals/youtube-venv, with
    google-api-python-client, google-auth-oauthlib and google-auth-httplib2;
  - an OAuth client of type "Desktop app" from Google Cloud Console (a project with the
    YouTube Data API v3 enabled), its JSON saved as ~/.config/nnfractals/youtube_client_secret.json.
The first upload opens the browser for Google's consent; the token is kept in
~/.config/nnfractals/youtube_token.json (only this user can read it).

Google's rule: an API project that has not passed YouTube's compliance audit uploads
videos as private (youtube.com/... support "yt_api_form" to request the audit).

Usage:
  youtube_upload.py --file VIDEO --title T [--description D] [--tags a,b] [--privacy private|unlisted|public]
                    [--category 1] [--thumbnail IMAGE]
  youtube_upload.py --auth [--lang en] [--browser "firefox --private-window"]
                                    (only the consent, asked again; to set it up or record the demo)
Prints one line each: STATUS <text>, PROGRESS <0..100>, DONE <video id>, WARN <text>, ERROR <text>.
"""
import argparse
import os
import random
import sys
import time

CONF = os.path.expanduser("~/.config/nnfractals")
SECRET = os.path.join(CONF, "youtube_client_secret.json")
TOKEN = os.path.join(CONF, "youtube_token.json")
# Upload, and set the thumbnail of what was uploaded (thumbnails.set accepts this scope).
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def say(kind, text=""):
    print(f"{kind} {text}".rstrip(), flush=True)


def fail(text):
    say("ERROR", text)
    sys.exit(1)


def credentials(fresh=False, lang=None, browser=None):
    """The saved login, refreshed if needed; Google's consent in the browser when there is
    none, or always with fresh=True (--auth). lang ("en") sets the consent screen's language;
    browser is a command the consent URL is appended to (e.g. "firefox --private-window")."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if not fresh and os.path.exists(TOKEN):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
        except Exception:
            creds = None
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_token(creds)
            return creds
        except Exception as e:  # revoked, or a 7-day "Testing" refresh token that expired
            say("STATUS", f"the saved Google login no longer works ({type(e).__name__}); asking again")
    if not os.path.exists(SECRET):
        fail(f"no OAuth client: save the JSON of a Desktop-app OAuth client (Google Cloud Console, YouTube Data API v3 enabled) as {SECRET}")
    flow = InstalledAppFlow.from_client_secrets_file(SECRET, SCOPES)
    say("STATUS", "waiting for Google's consent in the browser")
    extra = {"hl": lang} if lang else {}
    if browser:
        import shlex
        import webbrowser
        webbrowser.register("nnfractals-consent", None, webbrowser.GenericBrowser(shlex.split(browser) + ["%s"]))
        extra["browser"] = "nnfractals-consent"
    creds = flow.run_local_server(port=0, open_browser=True, authorization_prompt_message="",
                                  success_message="NNfractals can upload to YouTube now; you can close this tab.", **extra)
    save_token(creds)
    return creds


def save_token(creds):
    os.makedirs(CONF, exist_ok=True)
    fd = os.open(TOKEN, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(creds.to_json())


def clean(text, limit):
    # YouTube refuses "<" and ">" in titles and descriptions.
    return text.replace("<", "‹").replace(">", "›")[:limit]


def upload(a):
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    if not os.path.isfile(a.file):
        fail(f"no such video: {a.file}")
    yt = build("youtube", "v3", credentials=credentials(), cache_discovery=False)
    tags, total = [], 0
    for t in (x.strip() for x in a.tags.split(",")):
        if t and total + len(t) + 2 <= 480:  # YouTube allows ~500 characters of tags in all
            tags.append(t)
            total += len(t) + 2
    body = {
        "snippet": {"title": clean(a.title, 100) or "Fractal", "description": clean(a.description, 5000),
                    "tags": tags, "categoryId": str(a.category)},
        "status": {"privacyStatus": a.privacy, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(a.file, mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    say("STATUS", f"uploading {os.path.basename(a.file)} ({os.path.getsize(a.file) / 1e6:.0f} MB)")
    response, tries = None, 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                say("PROGRESS", str(int(status.progress() * 100)))
            tries = 0
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and tries < 8:
                tries += 1
                wait = min(64, 2 ** tries) + random.random()
                say("STATUS", f"YouTube answered {e.resp.status}, trying again in {wait:.0f} s")
                time.sleep(wait)
                continue
            fail(f"YouTube refused the upload: {e.resp.status} {e._get_reason() if hasattr(e, '_get_reason') else e}")
        except (OSError, ConnectionError, TimeoutError) as e:
            if tries < 8:
                tries += 1
                wait = min(64, 2 ** tries) + random.random()
                say("STATUS", f"network error ({type(e).__name__}), trying again in {wait:.0f} s")
                time.sleep(wait)
                continue
            fail(f"network error: {e}")
    vid = response.get("id")
    if not vid:
        fail(f"no video id in YouTube's answer: {response}")
    say("PROGRESS", "100")
    if a.thumbnail and os.path.isfile(a.thumbnail):
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(a.thumbnail)).execute()
            say("STATUS", "thumbnail set")
        except HttpError as e:  # custom thumbnails need a phone-verified channel
            say("WARN", f"thumbnail not set ({e.resp.status}): custom thumbnails need a verified channel")
    say("DONE", vid)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file")
    p.add_argument("--title", default="")
    p.add_argument("--description", default="")
    p.add_argument("--tags", default="")
    p.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])
    p.add_argument("--category", default="1")
    p.add_argument("--thumbnail")
    p.add_argument("--auth", action="store_true", help="only Google's consent in the browser, always asked again (set-up, or recording the demo for Google's verification)")
    p.add_argument("--lang", help="language of Google's consent screen, e.g. en (Google's verification wants the demo in English)")
    p.add_argument("--browser", help='browser command for the consent, e.g. "firefox --private-window" (a private window avoids '
                                     "Google's 500 error with many signed-in accounts, and keeps them out of a recording)")
    p.add_argument("--dry-run", action="store_true", help="no Google: fake progress and a fake id (tests; also env NNFRACTALS_YOUTUBE_DRY_RUN=1)")
    a = p.parse_args()
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
    except ImportError:
        fail("the uploader's libraries are missing: install them into ~/.local/share/nnfractals/youtube-venv")
    if a.auth:
        credentials(fresh=True, lang=a.lang, browser=a.browser)
        say("DONE", "authorised")
        return
    if not a.file:
        fail("--file is needed")
    if a.dry_run or os.environ.get("NNFRACTALS_YOUTUBE_DRY_RUN"):
        if not os.path.isfile(a.file):
            fail(f"no such video: {a.file}")
        say("STATUS", f"dry run: {os.path.basename(a.file)} as {a.privacy}, title {a.title!r}")
        for pct in (25, 50, 75, 100):
            say("PROGRESS", str(pct))
        say("DONE", "dryrun" + str(abs(hash(a.file)) % 10**6))
        return
    upload(a)


if __name__ == "__main__":
    main()
