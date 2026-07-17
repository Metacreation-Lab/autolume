# Autolume

Autolume is a cross-platform tool for real-time visual performances using StyleGAN.

For detailed documentation and demos look at the [Metacreation Lab Website](https://www.metacreation.net/autolume).

![Autolume Screen](docs/assets/overview-user-interface-01.jpg)

## Downloads

[➡️ Download for Windows and macOS](https://github.com/Metacreation-Lab/autolume/releases/latest)

Linux users need to run Autolume from sources following the [development instructions](#development-instructions).

## Examples

For examples of artworks created with Autolume see: https://www.metacreation.net/artworks

## Development instructions

`uv` is required for setting up the python environment ([installation link](https://docs.astral.sh/uv/#installation))

### Dependencies

#### Windows

- CUDA 12.8 ([download link](https://developer.nvidia.com/cuda-12-8-0-download-archive))
  - Minimum components: CUDA Development + CUDA Runtime
- Microsoft C++ Build Tools ([download link](https://download.visualstudio.microsoft.com/download/pr/13907dbe-8bb3-4cfe-b0ae-147e70f8b2f3/a3193e6e6135ef7f598d6a9e429b010d77260dba33dddbee343a47494b5335a3/vs_BuildTools.exe))
  - Minimum components: Desktop development with C++
- FFmpeg: `winget install Gyan.FFmpeg`

#### Linux (Ubuntu 24+)

- CUDA ([download link](https://developer.nvidia.com/cuda-12-8-0-download-archive))
  - Minimum components: CUDA Development + CUDA Runtime
- FFmpeg and PortAudio: `sudo apt install portaudio19-dev ffmpeg`

#### macOS

- FFmpeg and PortAudio: `brew install portaudio ffmpeg`
- Command Line Tools for Xcode: `xcode-select --install`

### Initial setup

Create the Python environment:

```bash
uv sync
```

### Running the program

Start the program
```bash
uv run main.py
```

### Create a release

Run the cross-platform release script on the OS you want to build for (PyInstaller
cannot cross-compile — a Windows build must run on Windows, etc.):
```bash
uv run release.py
```

ffmpeg/ffprobe are downloaded and bundled into the release automatically via
`ffmpeg-downloader`.

- **Windows / Linux:** output is the `dist/Autolume/` folder.
- **macOS:** output is `dist/Autolume.app`, code-signed automatically — see
  [macOS signing and notarization](#macos-signing-and-notarization).

Pass `--package` to additionally produce a distributable: an `.AppImage` on
Linux, a `.dmg` on macOS, or an Inno Setup installer plus a portable `.tar.xz`
archive on Windows (requires
[Inno Setup](https://jrsoftware.org/isdl.php) on the build machine, installable
with `winget install --id JRSoftware.InnoSetup.7 -e -s winget -i`). Use
`--package-only` to package an existing `dist/` output without rebuilding.

#### macOS signing and notarization

Gatekeeper only opens a downloaded app without friction when it is both
**signed** with a Developer ID and **notarized** by Apple (an automated
malware scan of the uploaded file — this applies to all distribution outside
the App Store, GitHub releases included). The release script handles both,
after a one-time setup.

**One-time setup**

1. *Signing certificate:* install a `Developer ID Application` certificate
   and its private key in your login keychain. Verify it is visible with:
   ```bash
   security find-identity -v -p codesigning
   ```
2. *Notarization credentials:* ask a team admin for an App Store Connect API
   key with the Developer role (App Store Connect → Users and Access →
   Integrations → Team Keys). You need three values from that page: the `.p8`
   key file, the Key ID, and the Issuer ID. Store them under the
   `autolume-notary` keychain profile:
   ```bash
   xcrun notarytool store-credentials autolume-notary \
     --key AuthKey_XXXXXX.p8 --key-id <key-id> --issuer-id <issuer-id>
   ```

**Release flow**

```bash
uv run release.py --package --notarize
```

- The build signs `Autolume.app` (hardened runtime + `entitlements.plist`)
  with the Developer ID certificate found in the keychain. If there is none,
  the script warns and falls back to an ad-hoc signature; the app then still
  runs locally, but downloads are blocked by Gatekeeper.
- `--package` wraps the app into a `.dmg` and signs it.
- `--notarize` submits the `.dmg` to Apple (usually takes a few minutes) and
  staples the resulting ticket, so users can open the app with a plain
  double-click. It can also be run on its own against an already-built `.dmg`
  — notarizing after the fact also unblocks copies that were published or
  downloaded earlier, since Gatekeeper checks Apple's servers online.

### Building documentation

```bash
uv run zensical build
```

The documentation will be output to `site`.

To serve the documentation locally:

```bash
uv run zensical serve
```

The documentation will be served at `http://127.0.0.1:8000/`.

## Credits

- Philippe Pasquier: Director of Metacreation Lab for Creative AI, Principal Investigator 
- Arshia Sobhan: Project management
- Lionel Ringenbach: Software development
- Michael Tjokrowardojo: Software development
- Jonas Kraasch: Project first-phase development
- Arthur Deleu: Project development
- Mahshid Jabari: DevOps
- Peter Chen: Contributions to the super-resolution module
- Manuel Bolduc: User testing
- Ahmed Abuzuraiq: Research associate

## Acknowledgment

StyleGAN3: https://github.com/NVlabs/stylegan3

GAN compression: https://github.com/lychenyoko/content-aware-gan-compression

GANSpace: https://github.com/harskish/ganspace

## License

Some of the dependencies fall under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).

## Project status

This code base is still open to changes and bugs might still appear. Please create issues and let us know so we can polish it for the final release!

As a dedicated research team behind Autolume, we emphasize that our system is a tool for exploration and innovation without direct support.
