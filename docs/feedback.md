# Feedback

We want to hear from you. Whether you have found a bug, want to request a
feature, or just want to share how you are using Autolume, there are two
ways to reach the development team.

## Feedback form

The quickest way is our feedback form. It takes any kind of feedback,
feature request, or issue report, and you do not need an account.

[Open the feedback form](https://docs.google.com/forms/d/e/1FAIpQLSe6ovWLmktE_AYGqxnSC_Ce1X6-A4X0_DAKeaEaej_RrBUgHQ/viewform)

You can also reach the form from inside Autolume with the **Feedback**
button in the toolbar.

## GitHub

If you prefer to track your report in public or already have a GitHub
account, open an issue on our repository. This is a good choice for bugs
with detailed reproduction steps or feature discussions.

[Open a GitHub issue](https://github.com/Metacreation-Lab/autolume/issues)

## Crash reports

When Autolume crashes, it offers to send a crash report to the development
team. Reports help us find and fix problems that only happen on real
setups. Sending is always your choice.

### What a report contains

- The crash error message and technical details of where it happened.
  This can include file paths from your system.
- The most recent application log (last portion of `logs/autolume.log`)
- The native crash log (`logs/crashes.log`)
- Your preferences file (`config.json`). It may include folder paths from
  your system, such as the location of your data folder.
- System information: Autolume version, operating system, Python version,
  and graphics hardware name
- A random installation id. It lets us tell whether ten reports come from
  one machine or ten machines. It is not linked to your name, email, or
  any account.

Reports never include your models, images, presets, or recordings.

### Choosing when to send

Open **Settings** and find **Crash reports**:

- **Ask before sending** (default): a dialog appears after a crash and you
  decide each time.
- **Always send**: reports upload automatically without asking.
- **Never send**: Autolume never sends anything.

If Autolume closes unexpectedly without showing a dialog, it offers to
send a report the next time you start it. You can add a short description
of what you were doing, which makes the report much more useful.
