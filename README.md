# Mouthpiece

Mouth words silently at your Mac's camera and Mouthpiece types them for you.

No sound needed. Everything runs on your own Mac, and nothing you say or record ever leaves your computer.

## What you need

* A Mac with Apple Silicon (M1 or newer) running macOS 13 or later
* A camera (the built in one is fine)
* A few GB of free space (the models alone are about 1.2 GB)

## Install

You only do this once. Open the **Terminal** app and run each command below.

**1. Install uv** (the tool that sets up Mouthpiece for you):

```bash
brew install uv
```

No Homebrew? Use this instead:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Download Mouthpiece:**

```bash
git clone https://github.com/colaway/mouthpiece.git
```

```bash
cd mouthpiece
```

**3. Set it up:**

```bash
uv sync
```

**4. Download the lip reading models** (about 1.2 GB, so this can take a few minutes):

```bash
./scripts/fetch_models.sh
```

**5. Check that everything is ready:**

```bash
uv run mouthpiece doctor
```

A green check means that part is fine. A warning tells you what to do next.

## Give your Mac permission

macOS asks before any app can use the camera, listen for a key, or type for you. These permissions go to the app you run Mouthpiece from (Terminal, iTerm or VS Code), not to Mouthpiece itself.

1. Open **System Settings**, then **Privacy & Security**.
2. Turn on your terminal app in each of these three lists:
   * **Camera**, so Mouthpiece can see your lips
   * **Input Monitoring**, so it knows when you hold the talk key
   * **Accessibility**, so it can type into other apps
3. Quit your terminal app completely and open it again.

If your terminal is not in the Camera list yet, this makes macOS ask for it:

```bash
uv run mouthpiece doctor --probe-camera
```

## Try it

```bash
uv run mouthpiece live
```

A window opens with your camera. Hold **Space**, mouth a short phrase, then let go. Mouthpiece shows what it read. Press **Q** to close the window.

## Use it every day

```bash
uv run mouthpiece app
```

A small mouth icon appears in the menu bar at the top of your screen. Click into any app where you want to type, then:

1. **Hold the right Option key** (⌥).
2. Mouth your words.
3. **Let go.** Mouthpiece reads your lips and types the result.

Quick taps of Option and shortcuts like Option+E still work normally.

### Two modes

**Command mode** (the default) listens for a fixed list of commands such as "copy that", "new line" or "scroll down". It is very accurate because it only picks from that list. A small popup shows its top 3 guesses. The first one is chosen after 2 seconds, or press **1**, **2** or **3** to pick, or **Esc** to cancel.

**Dictation mode** types whatever you say. Start in dictation mode with:

```bash
uv run mouthpiece app --mode dictation
```

You can also switch at any time by saying **"dictation mode"** or **"command mode"**, or from the menu bar icon.

### Dictation tips

* Say punctuation out loud: "period", "comma", "question mark", "exclamation mark".
* Say "new line" or "new paragraph" to start a new line.
* Say **"scratch that"** to remove what Mouthpiece just typed. Say it again to go further back.

For example, mouthing *"I think so" comma "but let me check" period* types:

> I think so, but let me check.

### Stopping

Click the menu bar icon and choose **Quit Mouthpiece**, or press **Control+C** in the terminal. **Pause camera** in the same menu turns the camera off without quitting.

## Teach it your lips

This step is optional, but it makes a big difference. The built in model can read most people a little. After it learns from your own recordings, it reads you much better.

**1. Record yourself:**

```bash
uv run mouthpiece record
```

A phrase appears on screen. Hold **Space**, mouth it silently, then let go. Then:

* **Enter** keeps the take
* **Backspace** tries it again
* **N** skips the phrase
* **Q** quits

One session asks for 100 takes. You can stop and come back later, and your progress is saved.

**2. See how much you have recorded:**

```bash
uv run mouthpiece dataset stats
```

**3. Train on your recordings** (needs at least 100 takes):

```bash
uv run mouthpiece train
```

Training takes a while. When it finishes, Mouthpiece only switches to the new version if it reads you better than the original. From then on it is used automatically.

Want to use the original model for a while? Start the app like this:

```bash
uv run mouthpiece app --pretrained
```

Your recordings are saved in the `data/personal` folder on your Mac. They are never uploaded, and that folder is never added to git.

## Commands you can say

These work in command mode.

**Editing:** new line, new paragraph, select all, copy that, paste it, cut that, undo that, redo that, delete the last word, bold that, italic that, save the file

**Fixing mistakes:** scratch that (removes the last thing Mouthpiece typed), delete that

**Punctuation:** period, comma, question mark, exclamation mark, open quote, close quote, open bracket, close bracket, press space

**Capitals for the next thing you say:** capital letter, all caps, no caps

**Moving around:** go to the start, go to the end, go back, go forward, move left, move right, move up, move down, page up, page down, scroll up, scroll down, select the line, select the word, select the next word, find on page

**Tabs and windows:** next tab, previous tab, open a new tab, close tab, close window, switch app, send it, press enter, cancel, press escape, press tab

**Your Mac:** open the browser, open my mail, open the terminal, search the web, volume up, volume down, mute, play music, pause music, take a screenshot, show the desktop, zoom in, zoom out

**Mouthpiece itself:** stop listening, start listening, dictation mode, command mode

**Quick replies** (typed as text): yes, no, okay, thank you, sounds good

In dictation mode, "scratch that", "delete that", "new line", "new paragraph" and the punctuation words also work as commands when you say them on their own.

## Try typing without the camera

This shows what Mouthpiece would type, without using the camera or typing anything:

```bash
uv run mouthpiece type --dry-run "I THINK SO" comma "BUT LET ME CHECK" period
```

## Settings

All settings live in one file: [`configs/mouthpiece.yaml`](configs/mouthpiece.yaml). Open it in any text editor. The ones people change most:

* `app.hotkey` is the key you hold to talk. Choices: `right_option`, `left_option`, `right_command`, `right_control`, `right_shift`, `fn`, or `f13` to `f19`.
* `app.mode` is the mode Mouthpiece starts in: `command` or `dictation`.
* `app.picker_timeout_s` is how many seconds the top 3 popup waits before choosing. Set it to `0` to always wait for you.
* `app.dictation_picker` set to `true` shows the top 3 popup in dictation mode too.
* `capture.camera_index` picks a different camera. `0` is the built in one.

## Troubleshooting

Run this first. It checks everything and tells you how to fix what it finds:

```bash
uv run mouthpiece doctor
```

**Nothing happens when I hold the key.** Turn on **Input Monitoring** for your terminal app, then quit and reopen the terminal.

**It reads my lips but doesn't type.** Turn on **Accessibility** for your terminal app, then quit and reopen the terminal.

**The camera won't open.** Turn on **Camera** for your terminal app. Also close other apps that might be using the camera, such as FaceTime or Zoom.

**The guesses are wrong a lot.** Face the camera in good light, keep your whole face in view, and mouth the words clearly. Command mode is much more accurate than dictation. Recording yourself and training (see "Teach it your lips") helps the most.

## Privacy

* Mouthpiece runs entirely on your Mac. There is no account, no server and no tracking.
* Camera frames are processed in memory and thrown away. Nothing is saved unless you record clips yourself, and those stay in the `data` folder on your Mac.
* The only time Mouthpiece uses the internet is when `fetch_models.sh` downloads the models during setup.

## How it works

1. Mouthpiece finds your face in each camera frame and crops a small picture of your mouth.
2. A lip reading model turns that sequence of mouth pictures into the most likely words.
3. In dictation mode, a language model helps it choose sensible sentences. In command mode, it simply picks the closest command from the list.
4. The result is typed into whatever app you are using.

The lip reading model is Auto AVSR, trained by its authors on thousands of hours of public videos of people talking. Mouthpiece can then train it a little further on your own recordings.

## For developers

Run the tests and checks:

```bash
uv run pytest
```

```bash
uv run ruff check .
```

```bash
uv run mypy src
```

Where things live:

* `src/mouthpiece/capture` reads the camera
* `src/mouthpiece/preprocess` finds the face and crops the mouth
* `src/mouthpiece/model` is the lip reading model and language model
* `src/mouthpiece/decode` turns model output into words
* `src/mouthpiece/record` and `src/mouthpiece/train` record your clips and train on them
* `src/mouthpiece/app` holds the windows and the menu bar app
* `src/mouthpiece/typing` turns results into typed text and key presses
* `configs` holds the settings file
* `tests` holds the test suite

Downloaded models go in `models` and recordings go in `data`. Git ignores both folders.

## License and credits

The Mouthpiece code is released under the [MIT License](LICENSE).

Files in `src/mouthpiece/model/` and `src/mouthpiece/decode/ctc_prefix.py` are adapted from [ESPnet](https://github.com/espnet/espnet) and the Auto AVSR code, and keep their Apache License 2.0 notices in the file headers.

The pretrained models are **not** part of this repository and are not covered by the MIT License. `scripts/fetch_models.sh` downloads them from their authors ([Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) and [auto_avsr](https://github.com/mpc001/auto_avsr)). They are licensed for **non commercial use only**. The MediaPipe face model and the Auto AVSR mean face file also keep their own terms.
