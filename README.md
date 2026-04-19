Hey everyone, I wanted to share a project I've been working on — a fully self-hosted, browser-based audio production tool built on top of the k2-fsa/OmniVoice diffusion model.
What it does:
It lets you turn a script into a finished, multi-speaker audio production — think podcast episodes, audiobook chapters, narrated videos — entirely on your own machine. No cloud, no subscriptions, no data leaving your computer.
Key features:

Voice cloning from a 3–10 second reference clip. Up to 4 independent speakers per project
Voice Designer — no reference audio? Describe a voice using attributes (gender, age, accent, pitch, style) and it generates one consistently across all your paragraphs
Multi-track timeline editor with waveform display, drag-to-reposition, trim handles, cut tool, ripple editing, and undo/redo
Media track for dropping in music, SFX or ambience alongside your voice content
Smart text parser — paste your script, it splits into paragraphs automatically. Use [Speaker 2]: to switch voices, [pause 2s] to insert timed silences
Episode save/load — saves everything: text, audio, timeline layout, voice settings, generation params
Pronunciation dictionary — fix proper nouns and technical terms once, applies to all generations
600+ language support out of the box, zero-shot

Hardware:
Runs on NVIDIA GPU, Apple Silicon (MPS), or CPU. Output is 24kHz WAV.
Tech stack:
Python/Flask backend, pure HTML/JS frontend (single file, no framework), OmniVoice diffusion model.
The whole thing runs locally — you just open the HTML file in a browser pointed at the Flask server. No install beyond pip install and pulling the model weights.
Happy to answer questions about the implementation. The voice design consistency system was probably the trickiest part to get right — making sure every paragraph sounds like the same person when you're designing a voice from scratch rather than cloning one.
