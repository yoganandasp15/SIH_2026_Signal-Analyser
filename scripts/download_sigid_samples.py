import os
import re
import subprocess
import json

TARGET_SIGNALS = [
    {
        "name": "RTTY",
        "url": "https://www.sigidwiki.com/wiki/RTTY",
        "expected_mod": "2-FSK (RTTY / Baudot)",
        "expected_baud": 45.45,
        "type": "Radioteletype"
    },
    {
        "name": "MFSK16",
        "url": "https://www.sigidwiki.com/wiki/MFSK",
        "expected_mod": "16-Tone MFSK",
        "expected_baud": 15.625,
        "type": "Multi-Frequency Shift Keying"
    },
    {
        "name": "WEFAX",
        "url": "https://www.sigidwiki.com/wiki/Weatherfax",
        "expected_mod": "Analog Facsimile (WEFAX / FM)",
        "expected_baud": None,
        "type": "Weather Facsimile"
    },
    {
        "name": "Morse_Code",
        "url": "https://www.sigidwiki.com/wiki/Morse_Code_(CW)",
        "expected_mod": "CW / Morse Code (OOK)",
        "expected_baud": None,
        "type": "Continuous Wave"
    },
    {
        "name": "CODAR",
        "url": "https://www.sigidwiki.com/wiki/CODAR",
        "expected_mod": "Pulsed / FMCW Radar",
        "expected_baud": None,
        "type": "Oceanographic HF Radar"
    }
]

output_dir = r"C:\ntro_signal_analyzer\verified_samples"
os.makedirs(output_dir, exist_ok=True)

print("[*] Inspecting SigIDWiki pages for audio links...")

for sig in TARGET_SIGNALS:
    name = sig["name"]
    url = sig["url"]
    print(f"\n[*] Fetching: {name} -> {url}")
    p = subprocess.run(["curl.exe", "-s", "-L", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", url], capture_output=True)
    html = p.stdout.decode("utf-8", errors="ignore")
    
    # Search for any mp3, wav, or ogg links anywhere in the page
    raw_links = re.findall(r'(/images/[^"\'<>\s]+\.(?:mp3|wav|ogg))', html, re.IGNORECASE)
    if not raw_links:
        # Check if there's a Special:FilePath or File: link
        raw_links = re.findall(r'href=[\'"]([^\'"]+Special:FilePath[^\'"]+)[\'"]', html, re.IGNORECASE)
    if not raw_links:
        raw_links = re.findall(r'href=[\'"]([^\'"]+File:[^\'"]+\.(?:mp3|wav|ogg))[\'"]', html, re.IGNORECASE)
    
    # Filter out thumbnail or icon images
    links = [l for l in raw_links if not any(x in l.lower() for x in ['thumb', 'icon', 'button'])]
    print(f"  Found links: {links[:4]}")
    if links:
        audio_link = links[0]
        if not audio_link.startswith("http"):
            audio_link = "https://www.sigidwiki.com" + audio_link
        
        ext = os.path.splitext(audio_link.split("?")[0])[1] or ".mp3"
        dest_raw = os.path.join(output_dir, f"{name}{ext}")
        dest_wav = os.path.join(output_dir, f"{name}.wav")
        
        if not os.path.exists(dest_raw) and not os.path.exists(dest_wav):
            print(f"  Downloading: {audio_link} -> {dest_raw}")
            subprocess.run(["curl.exe", "-s", "-L", "-A", "Mozilla/5.0", audio_link, "-o", dest_raw])
            print(f"  Downloaded size: {os.path.getsize(dest_raw) if os.path.exists(dest_raw) else 0} bytes")
        
        # Convert to WAV with ffmpeg if needed
        if os.path.exists(dest_raw) and not os.path.exists(dest_wav):
            print(f"  Converting to WAV: {dest_wav}")
            subprocess.run(["ffmpeg", "-y", "-i", dest_raw, "-ar", "44100", dest_wav], capture_output=True)
            print(f"  WAV created size: {os.path.getsize(dest_wav) if os.path.exists(dest_wav) else 0} bytes")
