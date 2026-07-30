# Voice assistant (Colab-ready)

Runs a spoken instruction in Hausa (or Yoruba/Igbo) through NCAIR's ASR
models, translates it to English, and hands it to an AI agent that reasons
about what needs to happen, picks the right tool(s), and carries them out
for real -- including compound instructions that need several steps.

## The agent architecture

The understanding/action step ("what does this instruction mean, and what
should be done about it") has two modes, switchable in the Streamlit
sidebar:

**Agent mode (default).** The translated English instruction is sent to an
LLM along with a set of 6 tools (real functions this project implements).
The model decides which tool(s) to call, in what order, with what inputs
-- it can break a single compound instruction into several tool calls on
its own. For example:

> "Schedule a meeting with John tomorrow at 3pm, remind me beforehand, and
> send him an email about it"

becomes 3 tool calls -- `schedule_event`, `set_reminder` (with a time it
works out on its own, e.g. 2:45pm), and `send_email` -- executed in
sequence, with a plain-language summary at the end. This is real reasoning
and planning, not a bigger keyword list.

**The default reasoning model is NCAIR's own N-ATLaS** -- using NCAIR's
own model for both the ears (ASR) and the thinking (N-ATLaS) of this
system, not an external provider, is the whole point. Claude and Gemini
are also wired in as backends (switchable in the sidebar) purely as
practical fallbacks if N-ATLaS turns out too slow on your hardware for a
live demo -- see "Setting up the agent" below for the honest tradeoffs of
each.

**Rules mode.** The original approach: a fixed keyword list decides one
intent (`intent.py`), then regex pulls out details (`entities.py`), then
exactly one action runs. Kept available specifically so the two approaches
can be compared side by side -- useful for explaining *why* a rule-based
approach struggles with compound or unusually-phrased requests, and what
an LLM-based approach does differently.

### The 6 tools available to the agent

| Tool | What it does | Key inputs | Real integration |
|---|---|---|---|
| `schedule_event` | Creates a Google Calendar event | summary, time, description | Google Calendar API |
| `set_reminder` | Creates a Calendar event with a popup notification | text, time | Google Calendar API |
| `send_email` | Sends a real email | contact (must be in contacts.json), message | Gmail API |
| `phone_call` | Logs a call request | contact, phone_number | None -- logged only, needs Twilio for a real call |
| `list_calendar_events` | Looks up upcoming events | time_min, time_max | Google Calendar API |
| `cancel_calendar_event` | Finds and deletes a matching event | search_text | Google Calendar API |

`list_calendar_events` and `cancel_calendar_event` are new -- they didn't
exist in the original 4-task version. They give the agent something
genuinely useful to do beyond the original scheduling/messaging tasks, and
are real integrations, not mocked. All three backends (NCAIR, Claude,
Gemini) call the exact same tool implementations in `agent_tools.py` --
only the reasoning model changes, not what the tools actually do.

### How natural-language time gets turned into a real date

Entity extraction and the agent tools both receive time as a loose string
("5pm", "tomorrow", "next monday") straight from the instruction. The
`dateparser` library (`calendar_action.py`) converts that string into an
actual Python datetime, using "prefer future dates" so "monday" means the
*next* Monday, not one that already passed. If it can't parse the string
at all, it falls back to one hour from now rather than failing the whole
action.

### Setting up the agent

**Backend 1: NCAIR's own N-ATLaS, running locally (default).** No API key
needed -- this uses NCAIR's own LLM
([NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS), an 8B-parameter
Llama-3 fine-tune for English/Hausa/Igbo/Yoruba). **Read this before
relying on it for a live demo:**

- The full model is ~20GB -- **not practical on a CPU-only laptop.** This
  project uses a quantized (~4.6GB) version instead, via `llama-cpp-python`.
- Even quantized, an 8B model on CPU is genuinely slow -- expect anywhere
  from 10 seconds to a couple of minutes per response. Test this on your
  actual machine well before demo day, not the night before.
- N-ATLaS has no built-in structured tool-calling like Claude or Gemini
  do. This project works around that with a hand-built protocol: the
  model is told about the tools in plain text and asked to always respond
  with a JSON object saying either "call this tool with these inputs" or
  "here's my final answer" (see `agent_ncair.py`). This is a reasonable
  approximation, not the same reliability as purpose-built tool-calling --
  expect it to occasionally not follow the format correctly, which is
  handled gracefully (you'll see the raw output rather than a crash) but
  is worth knowing about if it comes up in questions.

Setup:
```bash
# 1. Request access at https://huggingface.co/NCAIR1/N-ATLaS (same gated
#    flow as the ASR models -- may need approval, same as before)
# 2. Download the quantized version (~4.6GB, takes a while):
python download_models.py ncair-llm
# 3. Install llama-cpp-python -- this compiles from source:
xcode-select --install    # if you haven't already
pip install -r requirements-mac.txt
```

**Backend 2: Google Gemini (fallback, genuinely free tier, no card).**
Get a key at [aistudio.google.com](https://aistudio.google.com) with the
same Google account already used for Calendar/Gmail:

```bash
export GEMINI_API_KEY="..."
```

Honest note: this was built directly against Google's current published
docs, not tested against a live key (no network access to Google's API in
the environment this was built in) -- and Google's free-tier terms shift
fairly often, so it's worth checking
[ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)
yourself before relying on it.

**Backend 3: Claude (fallback, paid).** Anthropic doesn't have an ongoing
free tier -- new accounts get a small one-time trial credit, not
sustained free usage. Only worth setting up if you already have credits
or a paid account:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Pick the backend in the Streamlit sidebar (only shows up when "agent"
mode is selected). No other setup is needed for any backend beyond what
Google Calendar/Gmail already required.


## What's tested and verified

27 automated tests passing across `test_pipeline.py`, `test_agent_ncair.py`,
`test_agent_claude.py`, and `test_agent_gemini.py`, covering:

- Intent classification, entity extraction, action handlers (rules mode)
- All three agent backends' tool-calling loops -- mocked model responses
  confirm each correctly executes multiple tool calls, stops when no more
  tools are needed, and fails gracefully when a tool errors out (e.g.
  Google not configured)
- Task logging
- Full pipeline wiring in rules mode, tested end to end with text input

## What's NOT tested

- The ASR module (`asr.py`) needs internet access to download NCAIR's
  models -- not available in the environment this was built in.
- None of the three agents' real reasoning has been tested against a real
  model -- only each tool-execution loop itself was verified, using
  mocked responses standing in for what the real model would return. The
  actual quality of each model's step-by-step planning needs to be
  verified on your machine.

## Before this works for real


1. **Get a Hugging Face token** if NCAIR's models are gated (need to accept
   usage terms on the model's page first): https://huggingface.co/settings/tokens
2. **Verify the model IDs** in `asr.py` -- `NCAIR1/Igbo-ASR` is unconfirmed,
   check it exists on Hugging Face before relying on it.
3. **The entity extraction is regex-based, not a real NER model.** It's a
   fixed starting point (see `entities.py`) -- expect it to miss anything
   that doesn't roughly match "to/with [Name]", "that/saying [message]",
   common time phrases, or phone-number-shaped digit sequences. Treat it
   the same way as the intent patterns: test it, see what breaks, and
   tighten the patterns or replace with a real model later.
4. **schedule_event, set_reminder, and send_message now use real Google
   Calendar/Gmail** -- see the "Google Calendar & Gmail setup" section
   below for the one-time setup this needs. **phone_call is still logged
   only.** There's no free way to place a real phone call from Google's
   side -- that needs a paid telephony provider like Twilio (a phone
   number + per-minute cost). If you want real calls later, that's a
   separate setup, not covered here.

## Google Calendar & Gmail setup

This is a one-time setup on [Google Cloud Console](https://console.cloud.google.com)
(free, no billing needed for this usage level).

1. **Create a project.** Console → the project dropdown top-left → New
   Project → give it any name (e.g. "voice-assistant") → Create.
2. **Enable two APIs.** In the search bar at the top, search for and enable,
   one at a time: **Google Calendar API** and **Gmail API**.
3. **Configure the OAuth consent screen.** Left sidebar → APIs & Services →
   OAuth consent screen. Choose **External**, fill in an app name and your
   email for the required fields, and save through the steps. On the "Test
   users" step, **add your own Gmail address** -- this matters, since
   without it Google will refuse to let you log in during testing.
4. **Create credentials.** Left sidebar → APIs & Services → Credentials →
   Create Credentials → OAuth client ID → Application type: **Desktop app**
   → give it a name → Create.
5. **Download it.** Click the download icon next to the credential you just
   created, save it, and rename it to exactly `credentials.json`. Place it
   in this project folder (same folder as `app.py`).
6. **First run triggers login.** The first time you use schedule_event,
   set_reminder, or send_message, a browser tab opens asking you to log in
   and approve access. Since this app isn't verified by Google yet, you'll
   likely see an "unverified app" warning -- click **Advanced → Go to
   [app name] (unsafe)** to proceed; this is normal for apps you built
   yourself and haven't submitted for Google's review. After approving,
   a `token.json` is saved here so you won't need to log in again until it
   expires.
7. **Add real contacts.** `send_message` sends an actual email, so it needs
   a real address. Edit `contacts.json` and replace the example names with
   real ones, e.g. `{"musa": "actual.email@gmail.com"}`. Names are matched
   case-insensitively against whatever `entities.py` extracts as the contact.

**Note for Colab:** the login flow above opens a browser on your own
machine, which doesn't work inside Colab's VM. Google integration is
built to run from the local Streamlit webapp, not Colab -- use Colab for
testing ASR/translation/intent only, and the local webapp for anything
that needs to actually create a calendar event or send an email.

## Running as a local webapp (Streamlit)

**Before anything else:** the NCAIR models are gated on Hugging Face. Visit
the model page (e.g. https://huggingface.co/NCAIR1/Hausa-ASR), log in, and
request access. This step is required regardless of Colab vs. local -- it's
a permission on your Hugging Face account, not something the code controls.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# On an Intel Mac (torch has no builds newer than 2.2.2 for this platform):
pip install -r requirements-mac.txt

# On anything else (Apple Silicon, Linux, Windows with a modern torch):
pip install -r requirements.txt torch transformers

export HUGGINGFACE_TOKEN="your token here"
streamlit run app.py

## Running as a Desktop App

If you want to distribute the Voice Assistant as a standalone desktop application (no need to start Streamlit manually), you can use the provided `desktop_launcher.py` script and optionally package it with **PyInstaller**.

### 1. Run locally (development)
```bash
python desktop_launcher.py
```
This script starts the Streamlit server in a subprocess and automatically opens your default browser to `http://localhost:8501`. The app behaves exactly as when you run `streamlit run app.py`.

### 2. Build a standalone executable (macOS / Windows / Linux)
Install PyInstaller if you haven't already:
```bash
pip install pyinstaller
```
Create a single‑file binary:
```bash
pyinstaller --onefile --name voice_assistant desktop_launcher.py
```
The resulting executable (`dist/voice_assistant` on macOS/Linux or `dist\voice_assistant.exe` on Windows) can be distributed to users. When launched, it will:
- Start the Streamlit server bundled inside the binary.
- Open the default web browser pointing at the local app.
- Gracefully shut down the server on exit.

> **Note:** The packaged app still requires the Google OAuth `credentials.json` (if you want Calendar/Gmail integration) and a valid `GEMINI_API_KEY`/`ANTHROPIC_API_KEY` if you use those back‑ends. Place these files next to the executable or set the environment variables before launching.

### 3. Customising the bundle
If you want to include additional data files (e.g., a custom `contacts.json`), modify the PyInstaller command:
```bash
pyinstaller --onefile --add-data "contacts.json:." desktop_launcher.py
```
Adjust the `--add-data` argument for each file you need inside the bundle.

Now you have a fully functional desktop version of the Voice Assistant ready for demos or distribution!
```

Then open http://localhost:8501. Use **Option B (type text)** first to
confirm the non-ASR parts of the pipeline work, before testing **Option A
(record audio)**, which is the part that actually depends on Hugging Face
access and the heavier torch/transformers stack.

### Optional: download the models once, run fully offline after

By default, `asr.py` fetches each NCAIR model from Hugging Face the first
time it's used, then caches it in memory for that session only -- next time
you restart the app, it fetches again. If you'd rather download once and
never depend on the network again:

```bash
python download_models.py            # downloads all 3 languages
python download_models.py hausa       # or just one
```

This still requires Hugging Face access to be granted first -- it can't get
around that gate, it just avoids re-downloading every time. Models are saved
to `models/<language>/`. `asr.py` automatically checks that folder first and
only reaches out to Hugging Face if it's not there, so nothing else changes
in how you run the app.

## Running via Colab

See `project_colab.ipynb` -- upload this project as
`voice_assistant_starter.zip` in Step 1 of that notebook.



```bash
pip install -r requirements.txt
python -m pytest test_pipeline.py -v
```

## Project structure

```
pipeline.py       # orchestrator -- run_pipeline(lang, text=..., audio_path=..., mode="agent"|"rules", backend="ncair"|"claude"|"gemini")
agent_tools.py     # shared tool catalog + real execution logic (used by all 3 agent backends)
agent_ncair.py     # NCAIR backend (default) -- local N-ATLaS model, manual JSON tool-call protocol
agent_claude.py    # Claude backend (fallback) -- Anthropic tool-use API
agent_gemini.py    # Gemini backend (fallback) -- Google Interactions API, free tier
asr.py            # NCAIR ASR models, confidence scoring
translator.py     # English translation, both directions (deep-translator)
intent.py         # rules mode: keyword-based intent classification
entities.py       # rules mode: regex-based entity extraction
actions.py        # rules mode: the 4 action handlers
calendar_action.py # real Google Calendar integration (used by both modes)
gmail_action.py   # real Gmail send integration (used by both modes)
contacts.py       # name -> email lookup (contacts.json)
task_logger.py    # saves completed tasks to completed_tasks/
download_models.py # downloads ASR models + the N-ATLaS GGUF model
test_pipeline.py  # rules-mode + core tests
test_agent_ncair.py  # NCAIR agent loop tests (mocked, no GGUF file needed)
test_agent_claude.py # Claude agent loop tests (mocked, no API key needed)
test_agent_gemini.py # Gemini agent loop tests (mocked, no API key needed)
```

## Q&A prep -- one answer per likely question

**"Why this ASR model specifically?"** NCAIR's models are the only
publicly available ASR models fine-tuned specifically for Hausa, Yoruba,
and Igbo -- built on the Whisper architecture (an encoder-decoder
speech-to-text model), fine-tuned per language rather than using a single
generic multilingual model. General-purpose models like OpenAI's Whisper
support far fewer African languages well; NCAIR's models exist
specifically to close that gap.

**"Why translate to English at all -- why not process the local language
directly?"** The agent's reasoning (Claude) and the rules-mode classifier
both work with English text. Translation is the bridge: Hausa/Yoruba/Igbo
speech -> ASR -> local-language text -> **translated to English here** ->
agent reasons over the English text -> response generated in English ->
**translated back here** -> shown/spoken in the original language. Look at
`translator.py`'s two functions (`translate_to_english`,
`translate_from_english`) to point to exactly where this happens in the
code.

**"How does entity extraction differ between the two modes?"** In rules
mode, `entities.py` uses fixed regex patterns to pull out contact, time,
phone_number, and message *before* the intent runs, then hands them to a
single action. In agent mode, there's no separate extraction step --
Claude reads the whole instruction and decides what values to pass into
each tool call directly, including inferring values that were never stated
explicitly (like calculating a reminder time from a meeting time).

**"What happens with bad or incomplete input?"** In rules mode: if intent
classification fails, `NoMatchError` is caught in `pipeline.py` and a
"didn't recognize that instruction" response is returned. If a Google
action fails (e.g. missing credentials), every handler in `actions.py`
catches the error and returns a clear, honest message instead of crashing.
In agent mode: `_run_tool` in `agent.py` wraps every tool execution in a
try/except; on failure it returns a plain-text description of what went
wrong, which gets fed back to Claude so the final summary can mention it
honestly rather than claiming success.

**"What did each of us personally build?"** -- see
`group_presentation_prep.docx` for the full section-by-section breakdown;
match your section to the file(s) listed above so you can speak to your
specific code, not just the concept.

