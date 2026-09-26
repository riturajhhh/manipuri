import wave
import numpy as np
import io

# ============================================================
# Frame-based audio analysis constants
# ============================================================
FRAME_MS = 25      # 25ms frames
HOP_MS = 10        # 10ms hop (overlap)
VOICED_RMS_THRESHOLD = 0.05  # Frames above this RMS are considered voiced


def _compute_frames(audio_data, framerate):
    """Split audio into overlapping frames for per-frame analysis."""
    frame_len = int(framerate * FRAME_MS / 1000)
    hop_len = int(framerate * HOP_MS / 1000)
    
    if frame_len <= 0 or hop_len <= 0 or len(audio_data) < frame_len:
        return np.array([audio_data]) if len(audio_data) > 0 else np.array([])
    
    n_frames = 1 + (len(audio_data) - frame_len) // hop_len
    frames = np.zeros((n_frames, frame_len))
    
    for i in range(n_frames):
        start = i * hop_len
        frames[i] = audio_data[start:start + frame_len]
    
    return frames


def _spectral_features_frame(frame, framerate):
    """Compute spectral centroid, bandwidth, and rolloff for a single frame.
    
    Spectral centroid = weighted mean of frequencies by magnitude.
    Bandwidth = standard deviation of frequencies around centroid.
    Rolloff = frequency below which 85% of magnitude distribution is concentrated.
    """
    n = len(frame)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Apply Hanning window to reduce spectral leakage
    windowed = frame * np.hanning(n)
    
    # FFT magnitude spectrum (positive frequencies only)
    fft_mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / framerate)
    
    mag_sum = np.sum(fft_mag)
    if mag_sum < 1e-10:
        return 0.0, 0.0, 0.0
    
    centroid = np.sum(freqs * fft_mag) / mag_sum
    
    # Bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_mag) / mag_sum)
    
    # Rolloff (85%)
    cumsum = np.cumsum(fft_mag)
    rolloff_idx = np.where(cumsum >= 0.85 * mag_sum)[0]
    rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
    
    # Spectral Flux (requires previous frame, but here we can just return mag_sum for later diff, 
    # or just return the spectrum itself. A simpler approach is just returning mag_sum to diff later)
    flux_proxy = mag_sum
    
    return float(centroid), float(bandwidth), float(rolloff), float(flux_proxy)


def _estimate_pitch_autocorr(frame, framerate):
    """Estimate Pitch (F0) using autocorrelation."""
    if len(frame) == 0 or np.sum(frame**2) < 1e-6:
        return 0.0
    
    # Calculate autocorrelation
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]
    
    # Voice frequency bounds (50Hz to 500Hz)
    min_lag = int(framerate / 500.0)
    max_lag = int(framerate / 50.0)
    
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
        
    if min_lag >= max_lag:
        return 0.0
        
    # Search for peak in the valid lag range
    peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
    
    if corr[peak_lag] > 0.2 * corr[0]: # Threshold to ensure it's a prominent peak
        return framerate / float(peak_lag)
    return 0.0


def analyze_audio(file_bytes):
    """Analyze audio bytes of a WAV file using advanced frame-based DSP.
    
    Returns a dictionary of acoustic features:
    - rms_mean, rms_std: Frame-level RMS energy
    - zcr_mean, zcr_std: Frame-level Zero-Crossing Rate
    - spectral_centroid_mean, spectral_centroid_std: Voice brightness
    - spectral_bandwidth_mean, spectral_bandwidth_std: Voice spread/harshness
    - spectral_rolloff_mean, spectral_rolloff_std: High frequency content
    - pitch_mean, pitch_std, pitch_variability: Voice fundamental frequency
    - speaking_rate: Fraction of voiced frames
    """
    try:
        # Load wave file from bytes
        with wave.open(io.BytesIO(file_bytes), 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            if n_frames == 0:
                return {"status": "empty"}
                
            raw_frames = wf.readframes(n_frames)
            
            # Convert raw bytes to numpy array based on sample width
            if sampwidth == 2:
                audio_data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)
            elif sampwidth == 1:
                audio_data = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0
            else:
                # 24-bit or 32-bit floats
                audio_data = np.frombuffer(raw_frames, dtype=np.int8).astype(np.float32)
                
            # If multi-channel, merge to mono by averaging
            if n_channels > 1 and len(audio_data) > 0:
                audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
                
            if len(audio_data) == 0:
                return {"status": "empty"}
                
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)

            # Normalize audio signal to range [-1.0, 1.0]
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

            # ====== FRAME-BASED ANALYSIS ======
            frames = _compute_frames(audio_data, framerate)
            
            if len(frames) == 0:
                return {"status": "empty"}
            
            # Per-frame RMS
            frame_rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
            rms_mean = float(np.mean(frame_rms))
            rms_std = float(np.std(frame_rms))
            
            # Per-frame ZCR
            frame_zcr = np.array([
                np.sum(np.abs(np.diff(f >= 0))) / len(f) if len(f) > 1 else 0.0
                for f in frames
            ])
            zcr_mean = float(np.mean(frame_zcr))
            zcr_std = float(np.std(frame_zcr))
            
            # Per-frame spectral features (Centroid, Bandwidth, Rolloff, Flux Proxy)
            spectral_feats = np.array([_spectral_features_frame(f, framerate) for f in frames])
            sc_mean = float(np.mean(spectral_feats[:, 0]))
            sc_std = float(np.std(spectral_feats[:, 0]))
            bw_mean = float(np.mean(spectral_feats[:, 1]))
            bw_std = float(np.std(spectral_feats[:, 1]))
            ro_mean = float(np.mean(spectral_feats[:, 2]))
            ro_std = float(np.std(spectral_feats[:, 2]))
            
            # Spectral flux (difference between consecutive frames' proxy magnitude)
            if len(spectral_feats) > 1:
                fluxes = np.abs(np.diff(spectral_feats[:, 3]))
                flux_mean = float(np.mean(fluxes))
            else:
                flux_mean = 0.0
            
            # Dynamic Range
            dynamic_range = float(np.percentile(np.abs(audio_data), 95) - np.percentile(np.abs(audio_data), 5))
            
            # Speaking rate (fraction of voiced frames)
            voiced_count = int(np.sum(frame_rms > VOICED_RMS_THRESHOLD))
            speaking_rate = voiced_count / len(frames) if len(frames) > 0 else 0.0
            
            # Pitch estimation (Autocorrelation on voiced frames)
            voiced_frames = frames[frame_rms > VOICED_RMS_THRESHOLD]
            if len(voiced_frames) > 0:
                frame_pitch = np.array([_estimate_pitch_autocorr(f, framerate) for f in voiced_frames])
                valid_pitch = frame_pitch[frame_pitch > 0]
                if len(valid_pitch) > 0:
                    pitch_mean = float(np.mean(valid_pitch))
                    pitch_std = float(np.std(valid_pitch))
                    pitch_variability = pitch_std / (pitch_mean + 1e-5)
                else:
                    pitch_mean, pitch_std, pitch_variability = 0.0, 0.0, 0.0
            else:
                pitch_mean, pitch_std, pitch_variability = 0.0, 0.0, 0.0
                
            return {
                "status": "success",
                "rms": rms_mean,
                "rms_mean": rms_mean,
                "rms_std": rms_std,
                "zcr": zcr_mean,
                "zcr_mean": zcr_mean,
                "zcr_std": zcr_std,
                "dynamic_range": dynamic_range,
                "spectral_centroid_mean": sc_mean,
                "spectral_centroid_std": sc_std,
                "spectral_bandwidth_mean": bw_mean,
                "spectral_bandwidth_std": bw_std,
                "spectral_rolloff_mean": ro_mean,
                "spectral_rolloff_std": ro_std,
                "spectral_flux_mean": flux_mean,
                "speaking_rate": float(speaking_rate),
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "pitch_variability": pitch_variability,
                "framerate": framerate,
                "duration": n_frames / framerate
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def predict_audio_emotion(acoustic_features, target_classes):
    """Predict emotion probabilities based on advanced acoustic DSP features.
    
    Uses an enhanced multi-factor Arousal-Valence mapping with:
    - Frame-level RMS statistics
    - Pitch (F0) tracking and variability
    - Spectral Centroid, Bandwidth, and Rolloff
    - ZCR and Speaking Rate
    """
    rms_mean = acoustic_features.get("rms_mean", acoustic_features.get("rms", 0.3))
    rms_std = acoustic_features.get("rms_std", 0.1)
    zcr_mean = acoustic_features.get("zcr_mean", acoustic_features.get("zcr", 0.05))
    zcr_std = acoustic_features.get("zcr_std", 0.02)
    dynamic_range = acoustic_features.get("dynamic_range", 0.5)
    
    # Spectral
    sc_mean = acoustic_features.get("spectral_centroid_mean", 1000.0)
    bw_mean = acoustic_features.get("spectral_bandwidth_mean", 1000.0)
    ro_mean = acoustic_features.get("spectral_rolloff_mean", 2000.0)
    flux_mean = acoustic_features.get("spectral_flux_mean", 0.0)
    
    # Pitch & rate
    speaking_rate = acoustic_features.get("speaking_rate", 0.5)
    pitch_mean = acoustic_features.get("pitch_mean", 150.0)
    pitch_var = acoustic_features.get("pitch_variability", 0.0)
    
    # Normalizations for heuristics
    sc_norm = min(1.0, max(0.0, (sc_mean - 200.0) / 3800.0))
    bw_norm = min(1.0, max(0.0, bw_mean / 2500.0))
    ro_norm = min(1.0, max(0.0, ro_mean / 4000.0))
    pitch_norm = min(1.0, max(0.0, (pitch_mean - 50.0) / 450.0))
    flux_norm = min(1.0, max(0.0, flux_mean / 20.0)) # heuristic normalization for flux
    
    # === AROUSAL CALCULATION (multi-factor) ===
    # Arousal is highly sensitive to Pitch, Volume, and Rate
    arousal = (
        0.15 * min(1.0, rms_mean / 0.4) +          # Volume
        0.10 * min(1.0, rms_std / 0.15) +          # Volume dynamics
        0.10 * min(1.0, zcr_mean / 0.15) +         # Sibilance/sharpness
        0.15 * min(1.0, dynamic_range / 0.7) +     # Dynamic range
        0.10 * min(1.0, speaking_rate) +           # Speech activity
        0.05 * sc_norm +                           # Spectral brightness
        0.05 * bw_norm +                           # Spectral spread (harshness)
        0.15 * pitch_norm +                        # Pitch height (shouting/excitement)
        0.15 * flux_norm                           # Spectral flux (sudden changes)
    )
    arousal = min(1.0, max(0.0, arousal))
    
    # === VALENCE CALCULATION (multi-factor) ===
    # Positive valence relates to pitch expressiveness, brightness, and controlled stability
    valence = (
        0.20 * sc_norm +                           # Bright voice -> positive
        0.30 * min(1.0, pitch_var * 6.0) +         # Pitch expressiveness -> positive
        0.20 * (1.0 - min(1.0, zcr_std / 0.08)) +  # Stable ZCR -> controlled -> positive
        0.10 * (1.0 - ro_norm) +                   # Lower rolloff -> less harsh -> positive
        0.20 * min(1.0, speaking_rate * 1.2)       # Engaged speech -> positive
    )
    valence = min(1.0, max(0.0, valence))
    
    # Calibrated Arousal-Valence emotion coordinates
    emotion_av_coords = {
        'joy':          (0.70, 0.85),
        'sadness':      (0.15, 0.15),
        'anger':        (0.85, 0.10),
        'fear':         (0.80, 0.25),
        'surprise':     (0.80, 0.65),
        'disgust':      (0.50, 0.20),
        'tired':        (0.10, 0.30),
        'proud':        (0.60, 0.80),
        'calm':         (0.15, 0.75),
        'lonely':       (0.15, 0.15),
        'excited':      (0.90, 0.90),
        'love':         (0.50, 0.90),
        'neutral':      (0.40, 0.50),
        'trust':        (0.40, 0.70),
        'anticipation': (0.60, 0.55),
    }
    
    probs = []
    temperature = 0.18  # Sharpen distribution slightly more than before
    
    for cls in target_classes:
        cls_lower = cls.lower()
        coord = emotion_av_coords.get(cls_lower, (0.45, 0.50))
        dist = np.sqrt((arousal - coord[0])**2 + (valence - coord[1])**2)
        log_prob = -dist / temperature
        probs.append(log_prob)
    
    # Softmax normalization
    probs = np.array(probs)
    probs = probs - np.max(probs)
    probs = np.exp(probs)
    probs = probs / np.sum(probs)
    
    return probs


def fuse_text_and_audio(text_probs, audio_probs, text_weight=0.7, audio_features=None):
    """Perform confidence-adaptive Late Fusion of text and audio probabilities."""
    text_probs = np.array(text_probs, dtype=np.float64)
    audio_probs = np.array(audio_probs, dtype=np.float64)
    
    actual_text_weight = text_weight
    
    if audio_features is not None:
        rms = audio_features.get("rms_mean", audio_features.get("rms", 0.3))
        speaking_rate = audio_features.get("speaking_rate", 0.5)
        
        audio_quality = 0.5 * min(1.0, rms / 0.3) + 0.5 * min(1.0, speaking_rate)
        
        if audio_quality > 0.6:
            actual_text_weight = 0.60
        elif audio_quality < 0.3:
            actual_text_weight = 0.85
        else:
            actual_text_weight = text_weight
    
    fused = actual_text_weight * text_probs + (1.0 - actual_text_weight) * audio_probs
    
    fused = np.maximum(fused, 0.0)
    total = np.sum(fused)
    if total > 0:
        fused = fused / total
    
    return fused
