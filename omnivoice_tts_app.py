#!/usr/bin/env python3
"""
OmniVoice TTS Web Application
Voice cloning with multi-speaker support using k2-fsa/OmniVoice

Features:
- Voice Cloning with up to 4 speakers
- Paragraph management (split, move, insert, regenerate)
- Pause controls between paragraphs
- Media file insertion
- Episode save/load
- Pronunciation dictionary
- Full OmniVoice parameter controls
"""

import os
import sys
import json
import time
import base64
import shutil
import argparse
import threading
import re
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

import torch
import numpy as np
import soundfile as sf

# Optional Whisper
try:
    import whisper
except ImportError:
    whisper = None
    print("⚠ Whisper not available")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--device', type=str, default='auto',
                    help='Device: auto, cuda, mps, cpu')
parser.add_argument('--dtype', type=str, default='float16',
                    choices=['float16', 'bfloat16', 'float32'],
                    help='Model dtype')
args = parser.parse_args()

print("=" * 60)
print("  OmniVoice TTS Server")
print("=" * 60)

# Flask config
app = Flask(__name__, static_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# TensorFloat32
torch.set_float32_matmul_precision('high')

# Globals
model = None
model_lock = threading.Lock()
generation_lock = threading.Lock()
generation_busy = False
whisper_model = None
current_model_info = {'loaded': False, 'device': None, 'dtype': None}

# Voice samples: speaker_id -> {'audio_path': str, 'ref_text': str or None}
voice_samples = {1: None, 2: None, 3: None, 4: None}
VOICE_TEMP_DIR = 'voice_refs'

# Directories
EPISODES_DIR = 'episodes'
VOICES_DIR = 'voices'
MEDIA_DIR = 'media'
PRONUNCIATIONS_FILE = 'pronunciations.json'
MODELS_DIR = 'models/omnivoice'
SAMPLE_RATE = 24000  # OmniVoice outputs 24 kHz

for d in [EPISODES_DIR, VOICES_DIR, MEDIA_DIR, MODELS_DIR, VOICE_TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# Pronunciation dictionary
pronunciations = []
if os.path.exists(PRONUNCIATIONS_FILE):
    with open(PRONUNCIATIONS_FILE, 'r') as f:
        pronunciations = json.load(f)


def apply_pronunciations(text):
    for entry in pronunciations:
        if entry.get('original'):
            text = text.replace(entry['original'], entry.get('replace', ''))
    return text


# Whisper - used for /transcribe endpoint AND auto ref_text for voice cloning
if os.environ.get('DISABLE_WHISPER', '0') != '1' and whisper is not None:
    print("Loading Whisper...")
    try:
        whisper_model = whisper.load_model(
            "base",
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print("✓ Whisper loaded")
    except Exception as e:
        print(f"⚠ Whisper failed: {e}")

print()
print("=" * 60)
print(f"  READY - http://localhost:{args.port}")
print("=" * 60)
sys.stdout.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device():
    if args.device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return args.device


_DTYPE_MAP = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}


def load_model(device=None, dtype_str=None):
    """Load OmniVoice model. Returns True on success."""
    global model, current_model_info

    device = device or _resolve_device()
    dtype_str = dtype_str or args.dtype
    dtype = _DTYPE_MAP.get(dtype_str, torch.float16)

    # Already loaded with same config?
    if (model is not None
            and current_model_info['device'] == device
            and current_model_info['dtype'] == dtype_str):
        return True

    print(f"\nLoading OmniVoice...")
    print(f"  Device : {device}")
    print(f"  Dtype  : {dtype_str}")
    sys.stdout.flush()

    try:
        from omnivoice import OmniVoice

        if model is not None:
            del model
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        os.environ['HF_HOME'] = MODELS_DIR
        os.environ['HUGGINGFACE_HUB_CACHE'] = MODELS_DIR

        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
            cache_dir=MODELS_DIR,
        )

        current_model_info = {'loaded': True, 'device': device, 'dtype': dtype_str}
        print("✓ OmniVoice loaded")
        return True

    except Exception as e:
        print(f"✗ Load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Routes – model
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('omnivoice_index.html')


@app.route('/model_info')
def model_info():
    return jsonify({
        'loaded': model is not None,
        'model_name': 'OmniVoice (k2-fsa/OmniVoice)',
        'device': current_model_info.get('device'),
        'dtype': current_model_info.get('dtype'),
        'voices': {str(i): voice_samples[i] is not None for i in range(1, 5)},
    })


@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    data = request.get_json() or {}
    with model_lock:
        success = load_model(
            device=data.get('device'),
            dtype_str=data.get('dtype'),
        )
    if success:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Failed to load model'}), 500


@app.route('/unload_model', methods=['POST'])
def unload_model():
    global model, voice_samples, current_model_info
    with model_lock:
        if model is not None:
            del model
            model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        voice_samples = {1: None, 2: None, 3: None, 4: None}
        current_model_info = {'loaded': False, 'device': None, 'dtype': None}
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# Routes – voices
# ---------------------------------------------------------------------------

@app.route('/voice_status')
def voice_status():
    return jsonify({'voices': {str(i): voice_samples[i] is not None for i in range(1, 5)}})


@app.route('/create_voice', methods=['POST'])
def create_voice():
    """Upload a reference audio for a speaker slot (1-4).
    Saves as a normalised 24 kHz wav and optionally auto-transcribes with Whisper."""
    global voice_samples
    if 'reference_audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    speaker_id = int(request.form.get('speaker_id', 1))
    if speaker_id < 1 or speaker_id > 4:
        return jsonify({'error': 'Invalid speaker ID'}), 400

    audio_file = request.files['reference_audio']
    save_path = os.path.join(VOICE_TEMP_DIR, f'speaker_{speaker_id}.wav')

    try:
        audio_file.save(save_path)

        # Normalise to 24 kHz mono float32 wav
        audio_data, sr = sf.read(save_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sr != 24000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)
        audio_max = np.abs(audio_data).max()
        if audio_max > 0:
            audio_data = audio_data / max(audio_max, 1.0)

        # Trim to max 10 seconds — longer audio doesn't improve clone quality
        MAX_REF_SAMPLES = 10 * 24000
        if len(audio_data) > MAX_REF_SAMPLES:
            audio_data = audio_data[:MAX_REF_SAMPLES]
            print(f"  Speaker {speaker_id} ref audio trimmed to 10s")

        sf.write(save_path, audio_data.astype(np.float32), 24000)

        duration = len(audio_data) / 24000

        # Auto-transcribe with Whisper to get ref_text (improves cloning quality)
        ref_text = None
        if whisper_model is not None:
            try:
                result = whisper_model.transcribe(save_path, fp16=torch.cuda.is_available(), task="transcribe")
                ref_text = result['text'].strip()
                print(f"  Speaker {speaker_id} ref_text: {ref_text[:80]}")
            except Exception as e:
                print(f"  Whisper transcription failed: {e}")

        with model_lock:
            voice_samples[speaker_id] = {
                'audio_path': save_path,
                'ref_text': ref_text,
            }

        return jsonify({
            'success': True,
            'speaker_id': speaker_id,
            'duration': duration,
            'ref_text': ref_text,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_ref_text', methods=['POST'])
def update_ref_text():
    """Allow the user to manually set/correct the ref_text for a speaker."""
    global voice_samples
    data = request.get_json() or {}
    speaker_id = int(data.get('speaker_id', 1))
    ref_text   = data.get('ref_text', '').strip()
    if speaker_id < 1 or speaker_id > 4:
        return jsonify({'error': 'Invalid speaker ID'}), 400
    if voice_samples.get(speaker_id) is None:
        return jsonify({'error': 'No voice loaded for this speaker'}), 400
    voice_samples[speaker_id]['ref_text'] = ref_text or None
    return jsonify({'success': True})


@app.route('/clear_voice', methods=['POST'])
def clear_voice():
    global voice_samples
    data = request.get_json()
    speaker_id = int(data.get('speaker_id', 1))
    with model_lock:
        voice_samples[speaker_id] = None
    # Also clear any voice-design ref audio cached for this speaker
    vd_cache = os.path.join(VOICE_TEMP_DIR, f'vd_{speaker_id}.wav')
    if os.path.exists(vd_cache):
        try: os.remove(vd_cache)
        except: pass
    return jsonify({'success': True})


@app.route('/clear_vd_refs', methods=['POST'])
def clear_vd_refs():
    """Clear cached voice-design ref audio so next Generate All picks a fresh voice."""
    data = request.get_json() or {}
    speaker_ids = data.get('speaker_ids', [1, 2, 3, 4])
    for sid in speaker_ids:
        path = os.path.join(VOICE_TEMP_DIR, f'vd_{sid}.wav')
        if os.path.exists(path):
            try: os.remove(path)
            except: pass
    return jsonify({'success': True})


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if whisper_model is None:
        return jsonify({'error': 'Whisper not available'}), 503
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio'}), 400

    audio_file = request.files['audio']
    temp_path = 'temp_transcribe.wav'
    audio_file.save(temp_path)

    try:
        result = whisper_model.transcribe(temp_path, fp16=torch.cuda.is_available(), task="transcribe")
        os.remove(temp_path)
        return jsonify({'success': True, 'transcript': result['text'].strip()})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Routes – generation
# ---------------------------------------------------------------------------

@app.route('/generate_paragraph', methods=['POST'])
def generate_paragraph():
    global generation_busy

    with generation_lock:
        if generation_busy:
            return jsonify({'error': 'Generation in progress'}), 429
        generation_busy = True

    try:
        if not load_model():
            generation_busy = False
            return jsonify({'error': 'Model not loaded'}), 503

        data = request.get_json()
        text = data.get('text', '')
        speaker_id = int(data.get('speaker_id', 1))

        # OmniVoice generation parameters
        num_step           = int(data.get('diffusion_steps', 32))
        speed              = float(data.get('speed', 1.0))
        seed               = int(data.get('seed', 42))
        guidance_scale     = float(data.get('guidance_scale', 2.0))
        t_shift            = float(data.get('t_shift', 0.1))
        instruct           = data.get('instruct', '').strip() or None
        vd_ref_audio_id    = data.get('vd_ref_audio_id')

        if not text:
            generation_busy = False
            return jsonify({'error': 'No text'}), 400

        text = apply_pronunciations(text)

        # Voice design consistency: if a ref audio has been cached for this design speaker,
        # use it as ref_audio so all paragraphs sound like the same voice.
        vd_ref_path = None
        if vd_ref_audio_id and not voice_samples.get(speaker_id):
            cached = os.path.join(VOICE_TEMP_DIR, f'{vd_ref_audio_id}.wav')
            if os.path.exists(cached):
                vd_ref_path = cached

        # If this is a voice design first generation (no cache yet), use seed 42
        # so the initial voice is always reproducible and consistent.
        if vd_ref_audio_id and not vd_ref_path and not voice_samples.get(speaker_id):
            seed = 42

        # Reproducible seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # [pause] tags: split text, generate each segment, insert silence between them.
        pause_pattern = re.compile(r'\[pause(?:[:\s]+(\d+\.?\d*)\s*s?)?\]', re.IGNORECASE)
        split_pattern = re.compile(r'\[pause(?:[:\s]+\d+\.?\d*\s*s?)?\]', re.IGNORECASE)
        pause_durations = [float(m) if m else 1.0 for m in pause_pattern.findall(text)]
        parts = [p.strip() for p in split_pattern.split(text) if p.strip()]

        voice_info = voice_samples.get(speaker_id)

        start_time = time.time()
        all_audio = []

        for i, part in enumerate(parts):
            if part:
                gen_kwargs = dict(
                    text=part,
                    num_step=num_step,
                    speed=speed,
                    guidance_scale=guidance_scale,
                    t_shift=t_shift,
                )

                if voice_info is not None:
                    # Real voice clone — use ref_audio + ref_text, no instruct
                    gen_kwargs['ref_audio'] = voice_info['audio_path']
                    if voice_info.get('ref_text'):
                        gen_kwargs['ref_text'] = voice_info['ref_text']
                elif vd_ref_path:
                    # Cached voice design — clone the cached audio, drop instruct
                    # so OmniVoice treats it like a real clone for consistency
                    gen_kwargs['ref_audio'] = vd_ref_path
                    gen_kwargs.pop('instruct', None)
                elif instruct:
                    # First generation for this voice design — use instruct only
                    gen_kwargs['instruct'] = instruct

                with torch.no_grad():
                    audio_list = model.generate(**gen_kwargs)

                if audio_list and audio_list[0] is not None:
                    wav = audio_list[0]
                    # model.generate() may return a torch tensor or a numpy array
                    if isinstance(wav, np.ndarray):
                        wav_np = wav.flatten().astype(np.float32)
                    else:
                        if wav.dtype == torch.bfloat16:
                            wav = wav.to(torch.float32)
                        wav_np = wav.cpu().numpy().flatten().astype(np.float32)
                    all_audio.append(wav_np)

                    # Cache the first generated audio as the reference for future paragraphs
                    if vd_ref_audio_id and not voice_info and vd_ref_path is None and i == 0:
                        cache_path = os.path.join(VOICE_TEMP_DIR, f'{vd_ref_audio_id}.wav')
                        sf.write(cache_path, wav_np, SAMPLE_RATE)
                        vd_ref_path = cache_path  # use for subsequent [pause] segments

            # Insert silence after this segment if there's a [pause] tag following it
            if i < len(pause_durations):
                silence_samples = int(SAMPLE_RATE * pause_durations[i])
                all_audio.append(np.zeros(silence_samples, dtype=np.float32))

        if not all_audio:
            generation_busy = False
            return jsonify({'error': 'No audio generated'}), 500

        audio_np = np.concatenate(all_audio)
        elapsed  = time.time() - start_time
        duration = len(audio_np) / SAMPLE_RATE
        rtf      = elapsed / duration if duration > 0 else 0

        audio_b64 = base64.b64encode(audio_np.tobytes()).decode('utf-8')

        generation_busy = False
        return jsonify({
            'success': True,
            'audio': audio_b64,
            'sample_rate': SAMPLE_RATE,
            'duration': duration,
            'time': elapsed,
            'rtf': rtf,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        generation_busy = False
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Routes – pronunciations
# ---------------------------------------------------------------------------

@app.route('/pronunciations', methods=['GET'])
def get_pronunciations():
    return jsonify({'pronunciations': pronunciations})


@app.route('/pronunciations', methods=['POST'])
def save_pronunciations_endpoint():
    global pronunciations
    data = request.get_json()
    pronunciations = data.get('pronunciations', [])
    with open(PRONUNCIATIONS_FILE, 'w') as f:
        json.dump(pronunciations, f, indent=2)
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# Routes – media
# ---------------------------------------------------------------------------

@app.route('/upload_media', methods=['POST'])
def upload_media():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    f = request.files['file']
    filename = f.filename
    media_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(MEDIA_DIR, f"{media_id}_{filename}")
    f.save(save_path)

    try:
        audio_data, sr = sf.read(save_path)
        duration = len(audio_data) / sr
    except:
        duration = 0

    return jsonify({
        'success': True,
        'media_id': media_id,
        'filename': filename,
        'path': save_path,
        'duration': duration,
    })


@app.route('/media/<path:filename>')
def serve_media(filename):
    from flask import send_from_directory
    return send_from_directory(MEDIA_DIR, filename)


# ---------------------------------------------------------------------------
# Routes – episodes
# ---------------------------------------------------------------------------

@app.route('/episodes', methods=['GET'])
def list_episodes():
    episodes = []
    if os.path.exists(EPISODES_DIR):
        for name in os.listdir(EPISODES_DIR):
            meta_path = os.path.join(EPISODES_DIR, name, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                meta['id'] = name
                episodes.append(meta)
    return jsonify({'episodes': sorted(episodes, key=lambda x: x.get('created', ''), reverse=True)})


@app.route('/episodes/<episode_id>', methods=['GET'])
def load_episode(episode_id):
    episode_path = os.path.join(EPISODES_DIR, episode_id)
    if not os.path.exists(episode_path):
        return jsonify({'error': 'Not found'}), 404

    meta_path = os.path.join(episode_path, 'meta.json')
    data_path = os.path.join(episode_path, 'data.json')

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        with open(data_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return jsonify({'error': f'Failed to read episode data: {str(e)}'}), 500

    paragraphs = data.get('paragraphs', [])
    for i, para in enumerate(paragraphs):
        audio_file = os.path.join(episode_path, f'audio_{i}.wav')
        if os.path.exists(audio_file):
            try:
                audio_data, sr = sf.read(audio_file)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                para['audio'] = base64.b64encode(audio_data.astype(np.float32).tobytes()).decode('utf-8')
                para['has_audio'] = True
                para['duration'] = len(audio_data) / sr
            except Exception as e:
                print(f"Failed to load audio for paragraph {i}: {e}")
                para['audio'] = None
                para['has_audio'] = False
        else:
            para['audio'] = None
            para['has_audio'] = False

    return jsonify({
        'success': True,
        'meta': meta,
        'data': data,
        'paragraphs': paragraphs,
    })


@app.route('/episodes', methods=['POST'])
def save_episode():
    data = request.get_json()
    name = data.get('name', 'Untitled')
    paragraphs = data.get('paragraphs', [])
    settings = data.get('settings', {})

    episode_id = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower()).strip('_') or 'untitled'

    episode_path = os.path.join(EPISODES_DIR, episode_id)
    # Overwrite if already exists
    overwritten = os.path.exists(episode_path)
    if overwritten:
        shutil.rmtree(episode_path)
    os.makedirs(episode_path)

    total_duration = sum(p.get('duration', 0) for p in paragraphs)

    # Preserve original created date if this is an overwrite
    old_meta_path = os.path.join(EPISODES_DIR, episode_id, 'meta.json')
    original_created = datetime.now().isoformat()
    if os.path.exists(old_meta_path):
        try:
            with open(old_meta_path) as f:
                original_created = json.load(f).get('created', original_created)
        except Exception:
            pass

    meta = {
        'name': name,
        'created': original_created,
        'updated': datetime.now().isoformat(),
        'paragraph_count': len(paragraphs),
        'total_duration': total_duration,
        'settings': settings,
    }
    with open(os.path.join(episode_path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    save_paragraphs = []
    for i, para in enumerate(paragraphs):
        save_para = {
            'type': para.get('type', 'text'),
            'text': para.get('text', ''),
            'speaker_id': para.get('speaker_id', 1),
            'status': para.get('status', 'pending'),
            'pauseBefore': para.get('pauseBefore', 0),
            'pauseAfter': para.get('pauseAfter', 0.2),
            'media_filename': para.get('media_filename', ''),
            'duration': para.get('duration', 0),
        }
        if para.get('audio'):
            try:
                audio_bytes = base64.b64decode(para['audio'])
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                sf.write(os.path.join(episode_path, f'audio_{i}.wav'), audio_data, SAMPLE_RATE)
                save_para['has_audio'] = True
            except:
                save_para['has_audio'] = False
        else:
            save_para['has_audio'] = False
        save_paragraphs.append(save_para)

    with open(os.path.join(episode_path, 'data.json'), 'w') as f:
        json.dump({'paragraphs': save_paragraphs, 'settings': settings}, f, indent=2)

    return jsonify({'success': True, 'episode_id': episode_id, 'overwritten': overwritten})


@app.route('/episodes/<episode_id>', methods=['DELETE'])
def delete_episode(episode_id):
    episode_path = os.path.join(EPISODES_DIR, episode_id)
    if os.path.exists(episode_path):
        shutil.rmtree(episode_path)
        return jsonify({'success': True})
    return jsonify({'error': 'Not found'}), 404


# ---------------------------------------------------------------------------
# Routes – combine audio
# ---------------------------------------------------------------------------

@app.route('/combine_audio', methods=['POST'])
def combine_audio():
    data = request.get_json()
    paragraphs = data.get('paragraphs', [])

    all_audio = []

    for para in paragraphs:
        pause_before = float(para.get('pauseBefore', 0))
        pause_after  = float(para.get('pauseAfter', 0.2))

        if pause_before > 0:
            all_audio.append(np.zeros(int(SAMPLE_RATE * pause_before), dtype=np.float32))

        if para.get('audio'):
            try:
                audio_bytes = base64.b64decode(para['audio'])
                audio_data  = np.frombuffer(audio_bytes, dtype=np.float32)
                all_audio.append(audio_data)
            except:
                pass

        if pause_after > 0:
            all_audio.append(np.zeros(int(SAMPLE_RATE * pause_after), dtype=np.float32))

    if not all_audio:
        return jsonify({'error': 'No audio to combine'}), 400

    combined = np.concatenate(all_audio)
    duration = len(combined) / SAMPLE_RATE
    audio_b64 = base64.b64encode(combined.tobytes()).decode('utf-8')

    return jsonify({
        'success': True,
        'audio': audio_b64,
        'sample_rate': SAMPLE_RATE,
        'duration': duration,
    })


from flask import send_from_directory

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
