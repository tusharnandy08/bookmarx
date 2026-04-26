// Bridge: extract X auth cookies from the user's Comet profile by reusing
// fieldtheory's installed module. Prints {csrfToken, cookieHeader} as JSON.
//
// Usage: node tools/x_cookies.mjs [browserId]
//   browserId defaults to 'comet'. Other values: chrome, brave, chromium, helium, firefox.
//
// First run may trigger a macOS keychain prompt to unlock the browser's cookie store.
// If it fails, close the browser fully (so the Cookies sqlite is unlocked) and retry.

const FT_DIST_DIR = '/Users/tushar_nandy/.nvm/versions/node/v24.11.1/lib/node_modules/fieldtheory/dist';
const { extractChromeXCookies } = await import(`${FT_DIST_DIR}/chrome-cookies.js`);
const { getBrowser, browserUserDataDir } = await import(`${FT_DIST_DIR}/browsers.js`);

const browserId = process.argv[2] ?? 'comet';
const browser = getBrowser(browserId);
const userDataDir = browserUserDataDir(browser);

if (!userDataDir) {
  process.stderr.write(`No data dir known for ${browser.displayName} on this platform.\n`);
  process.exit(1);
}

try {
  const result = extractChromeXCookies(userDataDir, 'Default', browser);
  process.stdout.write(JSON.stringify(result));
} catch (err) {
  process.stderr.write(String(err?.message ?? err) + '\n');
  process.exit(1);
}
