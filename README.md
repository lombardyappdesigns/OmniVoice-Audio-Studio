# OmniVoice Audio Studio

OmniVoice Audio Studio is a fully self-hosted, browser-based audio production tool built on top of the k2-fsa/OmniVoice diffusion model.

# What it does:

It lets you turn a script into a finished, multi-speaker audio production — think podcast episodes, audiobook chapters, narrated videos — entirely on your own machine. No cloud, no subscriptions, no data leaving your computer.

## Key features:

Voice cloning from a 3–10 second reference clip. Up to 4 independent speakers per project
Voice Designer — no reference audio? Describe a voice using attributes (gender, age, accent, pitch, style) and it generates one consistently across all your paragraphs
Multi-track timeline editor with waveform display, drag-to-reposition, trim handles, cut tool, ripple editing, and undo/redo
Media track for dropping in music, SFX or ambience alongside your voice content
Smart text parser — paste your script, it splits into paragraphs automatically. Use [Speaker 2]: to switch voices, [pause 2s] to insert timed silences
Episode save/load — saves everything: text, audio, timeline layout, voice settings, generation params
Pronunciation dictionary — fix proper nouns and technical terms once, applies to all generations
600+ language support out of the box, zero-shot

## Hardware:

Runs on NVIDIA GPU, Apple Silicon (MPS), or CPU. Output is 24kHz WAV.

## Tech stack:

Python/Flask backend, pure HTML/JS frontend (single file, no framework), OmniVoice diffusion model.
The whole thing runs locally — you just open the HTML file in a browser pointed at the Flask server. No install beyond pip install and pulling the model weights.
Happy to answer questions about the implementation. The voice design consistency system was probably the trickiest part to get right — making sure every paragraph sounds like the same person when you're designing a voice from scratch rather than cloning one.

# User guide

**1 System Requirements & Installation**

### **Minimum Requirements**

|     |     |
| --- | --- |
| **Component** | **Requirement** |
| Operating System | Windows 10+, macOS 12+, Ubuntu 20+ |
| Python | 3.10 or higher |
| RAM | 8 GB minimum (16 GB recommended) |
| Storage | ~5 GB for model weights |
| GPU | Optional but strongly recommended |

### **Supported Hardware**

|     |     |     |
| --- | --- | --- |
| **Hardware** | **Mode** | **Performance** |
| NVIDIA GPU | CUDA | Best — RTF as low as 0.025 (40x real-time) |
| Apple Silicon | MPS | Good performance |
| CPU only | CPU | Works, but generation is slow |

## Installation

### 1. Set Up Virtual Environment

Ensure you have **Python 3.12** installed. A virtual environment keeps your project dependencies isolated from your global Python installation, preventing version conflicts.

Create and activate the virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate
```

> Once activated, your terminal prompt will be prefixed with `(venv)`, confirming the environment is active.

---

### 2. Install Dependencies

Install the required Python packages from the project's dependency file:

```bash
pip install -r requirements.txt
```

Next, install PyTorch with CUDA 12.8 support for GPU-accelerated inference. This version is required for optimal performance with compatible NVIDIA GPUs:

```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

> **No NVIDIA GPU?** See the [CPU mode tip](#tips) below to run on CPU instead.

---

### 3. Install FFmpeg

FFmpeg is required for audio processing and encoding. Install it using the package manager for your operating system:

| OS | Command |
|----|---------|
| **Windows** | `winget install ffmpeg` |
| **Mac** | `brew install ffmpeg` |
| **Linux** | `sudo apt update && sudo apt install ffmpeg` |

After installation, verify FFmpeg is accessible by running:

```bash
ffmpeg -version
```

---

### 4. Start the Backend

Launch the OmniVoice TTS server. This starts a local API that the frontend communicates with to generate speech:

```bash
python omnivoice_tts_app.py
```

The server will begin initializing and load the TTS model into memory. Wait until you see a message indicating it is ready before opening the frontend.

> **⚠️ First Run:** On first launch, the model (~3.3 GB) will be automatically downloaded from HuggingFace. This requires an active internet connection and may take several minutes depending on your connection speed. Subsequent launches will use the cached model and start much faster.

---

### 5. Open the Frontend

Once the backend is running, open the frontend interface using the link in the terminal window.

---

### Tips

> **Custom port:** Pass `--port 5000` when starting the backend to serve on a specific port:
> ```bash
> python omnivoice_tts_app.py --port 5000
> ```

> **CPU mode:** If you don't have a compatible NVIDIA GPU, pass `--device cpu` to force CPU inference (slower, but fully functional):
> ```bash
> python omnivoice_tts_app.py --device cpu
> ```




### 6. Loading the Model

The Load Model button pulses lavender until the model is loaded. Once loaded it turns amber and stays lit. The model only needs to be loaded once per session.

|     |     |     |
| --- | --- | --- |
| **Setting** | **Options** | **Recommendation** |
| Device | Auto, CUDA, MPS, CPU | Auto detects the best available hardware |
| Dtype | Float16, BFloat16, Float32 | Float16 is fastest on NVIDIA GPU. Use Float32 on CPU if quality is poor. |

**TIP:** Use Unload to free GPU VRAM between sessions if you need the memory for other tasks.


### 7. Setting Up Voices

OmniVoice Audio Studio supports two voice modes: Voice Clone (from a reference audio file) and Voice Designer (from text attributes). Up to 4 independent speakers can be configured. Each speaker uses one mode — they cannot be combined on the same speaker.

---

## **Voice Clone**

Clone any voice from a short reference audio clip.

**1\. Select a speaker slot** (1-4) by clicking its tab at the top of the Voice Setup card.

**2\. Click Choose Audio File** and select a WAV, MP3 or other audio file. 3-10 seconds of clean speech is ideal. Files longer than 10 seconds are automatically trimmed.

**3\. Click Create Voice.** If Whisper is installed, reference text is auto-transcribed for better quality. The speaker tab turns amber when ready.

**TIP:** Use a clean recording with no background music, minimal reverb, and consistent volume. A quiet room recording on a decent microphone works best.

**NOTE:** The Choose Audio File button pulses lavender until at least one voice is configured.

---

## **Voice Designer**

Design a voice from scratch using text attributes — no reference audio required.

|     |     |
| --- | --- |
| **Attribute** | **Options** |
| Gender | Male, Female |
| Age | Child, Teenager, Young Adult, Middle-aged, Elderly |
| Pitch | Very Low, Low, Moderate, High, Very High |
| Style | Whisper (or leave blank for normal) |
| English Accent | British, American, Australian, Indian, and 6 more |
| Chinese Dialect | 12 regional options |
| Notes | Free text — e.g. "warm, authoritative" |

**1\. Switch to the Voice Designer tab** in the Voice Setup card.

**2\. Select a speaker slot** and choose attributes from the dropdowns.

**3\. Click Apply to Speaker.** The description is built automatically from your selections.

**4\. Optionally click Save Design** to name and store the design for reuse in future projects.

**TIP:** The first paragraph generated with a Voice Design establishes the voice character. All subsequent paragraphs for that speaker clone that first audio for consistency across the whole project.

**NOTE:** Voice Clone and Voice Designer are mutually exclusive per speaker. If a clone is loaded, the designer is blocked for that speaker. Clear the clone first to switch modes.

**4 Generation Parameters**

Click the Generation Parameters collapsible to expand. These settings apply globally to all generations.

|     |     |     |
| --- | --- | --- |
| **Parameter** | **Default** | **Description** |
| Diffusion Steps | 32  | Higher = better quality, slower generation. Use 16 for quick preview, 64+ for final output. |
| Guidance Scale | 2.0 | How closely output follows the reference voice. Raise to 3-4 for stronger clone similarity. |
| Speed | 1.0x | Values >1 = faster speech. Values <1 = slower. Range: 0.5-2.0. |
| Seed | 42  | Global seed for reproducible output. Each paragraph also has its own seed slider. |
| Default Pause | 0.3s | Silence gap inserted between paragraphs. Can be overridden per paragraph. |
| Split By | Paragraph | How input text is split into paragraphs on Parse: by blank line, newline, or sentence. |

**5 Text Input & Parsing**

Paste your script into the Text Input area and click Parse Text to split it into paragraphs.

### **Special Syntax**

|     |     |
| --- | --- |
| **Syntax** | **Effect** |
| \[Speaker 2\]: | Assigns the following paragraph to Speaker 2's voice. Works with speakers 1-4. |
| \[pause\] | Inserts a pause of the default duration at that point in the audio. |
| \[pause 1.5s\] | Inserts a precise 1.5 second silence at that point in the audio. |

### **Split By Options**

|     |     |     |
| --- | --- | --- |
| **Mode** | **Splits On** | **Best For** |
| Paragraph | Blank lines between text blocks | Most scripts — natural paragraph breaks |
| Newline | Every line break | Poetry, dialogue, line-by-line scripts |
| Sentence | Each sentence ending in . ! ? | Fine-grained control, short audio bursts |

**NOTE:** Parse Text is disabled after audio has been generated. Use Reset All to start fresh with new text.

**6 Paragraph Management**

### **Paragraph Status**

|     |     |     |
| --- | --- | --- |
| **Status** | **Appearance** | **Meaning** |
| done | Amber border, warm dark background | Audio generated successfully |
| edited | Lavender border, cool dark background | Text changed after generation — needs regeneration |
| pending | Grey border | Not yet generated |
| generating | Lavender border | Currently being processed |

### **Per-Paragraph Controls**

|     |     |
| --- | --- |
| **Control** | **Function** |
| Drag handle (dots) | Drag to reorder paragraphs. Timeline updates automatically. |
| Speaker dropdown | Change which voice is used for this paragraph. |
| Generate button (music note) | Generate audio for this paragraph only. |
| Split button (scissors) | Split at the sentence midpoint into two paragraphs. |
| Delete button (x) | Remove this paragraph permanently. |
| Before / After inputs | Set the silence gap in seconds before and after this paragraph. |
| Seed slider | Default 42. Drag right for a different voice variation for this paragraph only. |

### **Inserting New Content**

Between every pair of paragraphs are + Text and + Media insert buttons:

- \+ Text — insert a new empty text paragraph at that position
- \+ Media — insert an audio file (music, SFX, ambience) at that position on the Media timeline track

### **Regenerate Pending**

The Regenerate Pending button appears automatically when any paragraph has been edited, split, or is waiting for generation. Clicking it generates all pending/edited paragraphs in sequence.

**7 Generating Audio**

**1\. Load the model** and set up at least one voice (clone or designer).

**2\. Parse your text** — click Parse Text to split into paragraphs.

**3\. Click Generate All** (or click the generate button on individual paragraphs). A progress bar shows generation status.

**4\. Review** — each paragraph shows a mini audio player once generated. Edit text and click Regenerate Pending to update specific paragraphs.

**5\. Combine All** — click Combine All to merge everything into a single audio file with all pauses applied.

**TIP:** When using Voice Designer, the first paragraph establishes the voice. All subsequent paragraphs for that speaker clone that first audio. Clicking Generate All always clears the cache and regenerates a fresh voice.

**TIP:** Individual paragraph seeds are shown as sliders. Default of 42 gives reproducible output. Change the seed to get a different voice variation for a specific paragraph.

**8 Timeline Editor**

The timeline shows two tracks — Voice (top) and Media (bottom). Each generated paragraph becomes a clip on the Voice track. Media files appear on the Media track and can be positioned freely.

### **Timeline Controls**

|     |     |
| --- | --- |
| **Action** | **Function** |
| Drag clip | Move a media clip to a new position. Voice clips stay sequential. |
| Drag clip left/right edge | Trim the clip non-destructively. Drag back to restore original. |
| Shift + click clip | Multi-select clips. |
| Click inside clip | Seek the playhead to that position in the combined audio. |
| Alt + click inside clip | Cut the clip at that point, splitting it in two. |
| Ripple All button | When on, moving a clip shifts all following clips to maintain gaps. |
| Ctrl + Z | Undo last timeline operation. |
| Ctrl + Y | Redo. |
| \- / + buttons | Zoom timeline out / in. |

**TIP:** Trimming is non-destructive — the original audio is always preserved. Drag the edge back to restore.

**TIP:** Zoom in for precise trimming of silence at the start or end of a clip. Zoom out for an overview of the full project layout.

**9 Combined Output & Export**

**1\. Combine All** — merges all paragraph audio, applying all pauses and timeline positions into a single audio buffer.

**2\. Preview** — the combined audio player lets you listen to the full output. The currently playing paragraph is highlighted in the Paragraphs section.

**3\. Download WAV** — exports the final output as a 24kHz 32-bit float WAV file, ready for use in any DAW or podcast host.

**TIP:** If you adjust clip positions in the timeline after combining, click Combine All again to rebuild the output with the updated layout.

**10 Episodes — Save & Load**

Episodes save your entire project: all text, generated audio, timeline layout, voice settings, and generation parameters. Use the Episodes button in the header to open the Episodes panel.

### **Saving an Episode**

**1\. Click Save Episode** or open the Episodes panel and click the Save tab.

**2\. Enter a name** for your episode and click Save.

**3\. Episodes are stored** on the server in the episodes/ folder. Saving over an existing name overwrites it.

### **Loading an Episode**

**1\. Click the Episodes button** in the header toolbar.

**2\. Find your episode** in the list and click Load.

**3\. All paragraphs, audio and settings** are restored. Voice files must be re-uploaded if you want to regenerate audio.

**NOTE:** Voice reference files are not saved in the episode — only the generated audio. Keep your original reference audio clips if you need to regenerate with the same cloned voice.

**11 Pronunciation Dictionary**

Define word substitutions that apply automatically before every generation. Useful for proper nouns, acronyms, or technical terms the model may mispronounce.

**1\. Click Pronunciations** in the header toolbar.

**2\. Click + Add Entry** and type the original word and its replacement.

**3\. Click Save.** Substitutions apply to all future generations automatically.

### **Examples**

|     |     |
| --- | --- |
| **Original** | **Replacement** |
| AWS | Amazon Web Services |
| SQL | sequel |
| Dr. | Doctor |
| APIs | A P I s |

**12 Tips & Best Practices**

### **Voice Cloning Quality**

- Use 5-10 seconds of clean, clear speech with no background noise
- Avoid music, reverb, or processing on the reference recording
- Match the reference language to your target text for best results
- Raise Guidance Scale to 3-4 for stronger speaker similarity
- Use Diffusion Steps 64+ for final quality output

### **Performance**

- Use Float16 dtype on NVIDIA GPU for best speed
- Use Steps 16 for quick previews during drafting
- Keep paragraphs under ~100 words each for consistent pacing
- Unload the model to free GPU VRAM between sessions

### **Writing for TTS**

- Write out numbers: "forty-two" not "42"
- Use the Pronunciation Dictionary for acronyms and technical terms
- Add \[pause 1s\] for dramatic beats and breathing room
- Use \[Speaker 2\]: for natural multi-voice dialogue
- Split very long paragraphs using the Split button for better pacing control

### **Troubleshooting**

|     |     |
| --- | --- |
| **Problem** | **Solution** |
| Model won't load | Check internet connection on first run. Try --device cpu if GPU fails. |
| Robotic or clipping audio | Reduce Guidance Scale. Ensure reference audio is clean and under 10 seconds. |
| Different voice between paragraphs | Use Generate All (not individual generates) to ensure consistent voice design caching. |
| Parse Text button greyed out | Audio already generated. Click Reset All to start fresh with new text. |
| Episodes not saving | Check the episodes/ folder exists and Flask has write permissions. |
| Audio out of sync on timeline | Click Combine All to rebuild the combined audio from current clip positions. |
| No audio after loading episode | Voice files are not stored in episodes. Re-upload your reference audio clips and recreate the voices. |

