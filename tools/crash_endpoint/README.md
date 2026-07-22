# Crash report endpoint

Google Apps Script web app receiving Autolume crash reports. Reports are
JSON POSTs (`{"token", "meta", "zip_b64"}`) sent by `utils/crash_report.py`.
Everything lives in one self-contained Drive folder: the script project
plus, created on demand on the first report, the report spreadsheet
(`autolume-crash-reports`, header row included) and one zip subfolder
per app version. The folder can be moved or renamed freely.

## Deploying (lab Google account)

1. In Drive, create a folder `autolume-crash-reports`. Inside it, New >
   More > Google Apps Script, name the project `autolume-crash-endpoint`.
2. Copy `.env.example` to `.env` at the repo root (gitignored) and set
   `AUTOLUME_CRASH_SCRIPT_ID` to the script id from the editor URL
   (`script.google.com/home/projects/SCRIPT_ID/edit`).
3. Run `tools/crash_endpoint/deploy.sh`. It pushes the code, creates the
   web app deployment, and writes `AUTOLUME_CRASH_REPORT_URL` and a
   generated `AUTOLUME_CRASH_REPORT_TOKEN` into `.env`.
4. Finish in the editor, following the deploy output: Run the `setup`
   function once, accepting the authorization prompt (grants Drive and
   Sheets access and creates the report spreadsheet beside the project),
   then set the Script Property `AUTOLUME_TOKEN` (Project Settings) to
   the printed token. Optional properties: `ALERT_EMAIL` and `MAX_ZIP_MB`
   (accepted zip size, default 1). Property changes apply immediately,
   no redeployment needed.

Rotating the token invalidates reports from builds that shipped the old
value.

## Updating the deployed script

Edit `endpoint.gs` (and `appsscript.json` if needed), then run:

    tools/crash_endpoint/deploy.sh

It pushes the local files, creates a new script version, and points the
existing web app deployment at it. The `/exec` URL stays stable; builds
in the wild keep working. The only configuration it reads is
`AUTOLUME_CRASH_SCRIPT_ID`; no clasp state is kept on disk.

One-time setup per machine and Google account:

1. Install Node.js (the script runs clasp via `npx`).
2. Enable the Apps Script API for the account:
   https://script.google.com/home/usersettings
3. `npx --yes @google/clasp login`
