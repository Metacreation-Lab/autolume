// Autolume crash report endpoint.
// Deploy as a web app: execute as me, accessible to anyone.
// Storage is created on demand beside this script in its Drive folder:
// the report spreadsheet and one zip subfolder per app version. If the
// script sits in the My Drive root, an 'autolume-crash-reports' folder
// is used instead.
// Script Properties (never commit values):
//   AUTOLUME_TOKEN  comma-separated accepted shared secrets. Builds ship the
//                   one in .env (AUTOLUME_CRASH_REPORT_TOKEN); rotation = add
//                   the new token here, drop the old once its traffic dies.
//   ALERT_EMAIL     optional, sends a mail per report when set
//   MAX_ZIP_MB      optional, accepted zip size limit (default 1)

var FOLDER_NAME = 'autolume-crash-reports';
var SHEET_NAME = 'autolume-crash-reports';
var SHEET_HEADER = ['Date', 'Version', 'OS', 'OS version', 'GPU', 'Type',
                    'Exception', 'Install id', 'Comment', 'Zip'];

// Run once from the editor after the first deploy: triggers the OAuth
// consent for the script's scopes and provisions the storage so it shows
// up in Drive immediately.
function setup() {
  sheet_(rootFolder_());
}

function doPost(e) {
  try {
    var props = PropertiesService.getScriptProperties();
    var body = JSON.parse(e.postData.contents);
    var tokens = String(props.getProperty('AUTOLUME_TOKEN') || '')
        .split(',').map(function (t) { return t.trim(); }).filter(String);
    if (!body.token || tokens.indexOf(body.token) === -1) {
      return respond_('forbidden');
    }
    var maxZipBytes = (Number(props.getProperty('MAX_ZIP_MB')) || 1) * 1024 * 1024;
    if (!body.zip_b64 || body.zip_b64.length > maxZipBytes * 1.4) {
      return respond_('too_large');
    }
    var meta = body.meta || {};
    var version = String(meta.app_version || 'unknown');
    var bytes = Utilities.base64Decode(body.zip_b64);
    var name = Utilities.formatDate(new Date(), 'UTC', 'yyyyMMdd-HHmmss')
        + '-' + String(meta.install_id || 'anon').slice(0, 8) + '.zip';
    var lock = LockService.getScriptLock();
    lock.waitLock(10000);
    var folder, sheet;
    try {
      var root = rootFolder_();
      folder = subfolder_(root, version);
      sheet = sheet_(root);
    } finally {
      lock.releaseLock();
    }
    var file = folder
        .createFile(Utilities.newBlob(bytes, 'application/zip', name));

    sheet.appendRow([
      new Date(), safeCell_(version), safeCell_(meta.os), safeCell_(meta.os_version),
      safeCell_(meta.gpu), safeCell_(meta.crash_type),
      safeCell_(String(meta.exception || '').slice(0, 500)), safeCell_(meta.install_id),
      safeCell_(meta.comment), file.getUrl(),
    ]);

    var alertEmail = props.getProperty('ALERT_EMAIL');
    if (alertEmail) {
      MailApp.sendEmail(alertEmail, 'Autolume crash report (' + version + ')',
          'Type: ' + String(meta.crash_type || '') + '\n' + String(meta.exception || '').slice(0, 1000)
          + '\n\n' + file.getUrl());
    }
    return respond_('ok');
  } catch (err) {
    return respond_('error: ' + String(err).slice(0, 200));
  }
}

// Drive name lookups include trashed items; using one would silently
// grow new reports inside the trash after a delete-and-reprovision.
function firstUntrashed_(iter) {
  while (iter.hasNext()) {
    var item = iter.next();
    if (!item.isTrashed()) {
      return item;
    }
  }
  return null;
}

function rootFolder_() {
  var parents = DriveApp.getFileById(ScriptApp.getScriptId()).getParents();
  if (parents.hasNext()) {
    var parent = parents.next();
    if (!parent.isTrashed() && parent.getId() !== DriveApp.getRootFolder().getId()) {
      return parent;
    }
  }
  return subfolder_(DriveApp.getRootFolder(), FOLDER_NAME);
}

function subfolder_(parent, name) {
  var existing = firstUntrashed_(parent.getFoldersByName(name));
  return existing !== null ? existing : parent.createFolder(name);
}

function sheet_(root) {
  // Match by name and type: the script project may share the folder name.
  var files = root.getFilesByName(SHEET_NAME);
  while (files.hasNext()) {
    var file = files.next();
    if (!file.isTrashed() && file.getMimeType() === MimeType.GOOGLE_SHEETS) {
      return SpreadsheetApp.openById(file.getId()).getSheets()[0];
    }
  }
  var ss = SpreadsheetApp.create(SHEET_NAME);
  var sheet = ss.getSheets()[0];
  sheet.appendRow(SHEET_HEADER);
  sheet.getRange(1, 1, 1, SHEET_HEADER.length).setFontWeight('bold');
  sheet.setFrozenRows(1);
  DriveApp.getFileById(ss.getId()).moveTo(root);
  return sheet;
}

// Report fields are attacker-controlled text. A leading formula character
// would make Sheets evaluate the cell; the apostrophe escape stores it as
// literal text without changing what the cell displays.
function safeCell_(value) {
  var s = String(value || '');
  return /^[=+\-@\t\r]/.test(s) ? "'" + s : s;
}

function respond_(status) {
  return ContentService.createTextOutput(JSON.stringify({status: status}))
      .setMimeType(ContentService.MimeType.JSON);
}
