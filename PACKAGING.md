# macOS Codesigning and Notarization (Optional)

For distribution outside the App Store, Apple recommends codesigning and notarizing macOS executables so users can run them without Gatekeeper warnings.

## Build first

```bash
pyinstaller -y --clean Jelnyelv.spec
```

Output: `dist/Jelnyelv` (onefile executable)

## Prerequisites

- Apple Developer account
- Developer ID Application certificate
- Notarization credentials (App-specific password)

## Codesigning

After building with PyInstaller:

```bash
codesign --deep --force --sign "Developer ID Application: Your Name (TEAM_ID)" dist/Jelnyelv
```

Verify:

```bash
codesign -vv dist/Jelnyelv
```

## Notarization

1. Create a zip for submission:

   ```bash
   ditto -c -k --keepParent dist/Jelnyelv dist/Jelnyelv.zip
   ```

2. Submit to Apple:

   ```bash
   xcrun notarytool submit dist/jelnyelv.zip \
     --apple-id "your@email.com" \
     --team-id "TEAM_ID" \
     --password "app-specific-password" \
     --wait
   ```

3. Staple the notarization ticket:

   ```bash
   xcrun stapler staple dist/Jelnyelv
   ```

## CI Integration

These steps are documented but not automated in CI. To add them to GitHub Actions:

- Store certificates and credentials as secrets
- Add a post-build step for macOS that runs the above commands
- Ensure the notarytool wait completes before stapling
