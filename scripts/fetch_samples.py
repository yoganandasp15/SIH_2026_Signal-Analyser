"""
SigIDWiki Sample Downloader and Benchmark Collector
===================================================
Fetches verified audio samples (.wav / .mp3) and ground-truth metadata
directly from the Signal Identification Guide wiki (sigidwiki.com).
Converts all downloaded mp3 files to standard WAV PCM for bit-exact DSP testing.
"""

import os
import re
import json
import subprocess
import urllib.request
from typing import Dict, Any, List

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

TARGET_PAGES = [
    {
        "name": "APRS",
        "url": "https://www.sigidwiki.com/wiki/Automatic_Packet_Reporting_System_(APRS)",
        "expected_mod": "AFSK / 2-FSK (Bell 202)",
        "expected_baud": 1200,
        "type": "Digital Packet Data"
    },
    {
        "name": "POCSAG",
        "url": "https://www.sigidwiki.com/wiki/POCSAG",
        "expected_mod": "2-FSK / FSK",
        "expected_baud": 1200,
        "type": "Paging / Digital Data"
    },
    {
        "name": "FT8",
        "url": "https://www.sigidwiki.com/wiki/FT8",
        "expected_mod": "8-FSK / M-FSK",
        "expected_baud": 6.25,
        "type": "Weak Signal Amateur"
    },
    {
        "name": "AIS",
        "url": "https://www.sigidwiki.com/wiki/Automatic_Identification_System_(AIS)",
        "expected_mod": "GMSK / FSK",
        "expected_baud": 9600,
        "type": "Maritime TDMA Data"
    },
    {
        "name": "DMR",
        "url": "https://www.sigidwiki.com/wiki/Digital_Mobile_Radio_(DMR)",
        "expected_mod": "4-FSK / TDMA",
        "expected_baud": 4800,
        "type": "Digital Mobile Radio"
    },
    {
        "name": "D-STAR",
        "url": "https://www.sigidwiki.com/wiki/D-STAR",
        "expected_mod": "GMSK / Continuous Phase",
        "expected_baud": 4800,
        "type": "Digital Amateur Voice/Data"
    },
    {
        "name": "STANAG_4285",
        "url": "https://www.sigidwiki.com/wiki/STANAG_4285",
        "expected_mod": "8-PSK / PSK",
        "expected_baud": 2400,
        "type": "Military HF Data"
    },
    {
        "name": "MFSK16",
        "url": "https://www.sigidwiki.com/wiki/MFSK16",
        "expected_mod": "16-FSK / M-FSK",
        "expected_baud": 15.625,
        "type": "Multi-Frequency Shift Keying"
    },
    {
        "name": "NAVTEX",
        "url": "https://www.sigidwiki.com/wiki/NAVTEX",
        "expected_mod": "2-FSK / SITOR-B",
        "expected_baud": 100,
        "type": "Maritime Navigational Telex"
    },
    {
        "name": "RTTY",
        "url": "https://www.sigidwiki.com/wiki/Radioteletype_(RTTY)",
        "expected_mod": "2-FSK / RTTY",
        "expected_baud": 45.45,
        "type": "Radioteletype"
    },
    {
        "name": "PSK31",
        "url": "https://www.sigidwiki.com/wiki/PSK31",
        "expected_mod": "BPSK / PSK",
        "expected_baud": 31.25,
        "type": "Phase Shift Keying"
    },
    {
        "name": "WEFAX",
        "url": "https://www.sigidwiki.com/wiki/Weather_Fax_(WEFAX)",
        "expected_mod": "FM / Analog Facsimile",
        "expected_baud": None,
        "type": "Weather Facsimile"
    }
]


def fetch_page_audio_and_meta(page_info: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    url = page_info["url"]
    name = page_info["name"]
    print(f"[*] Querying SigIDWiki: {name} ({url})...")
    
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        
    audio_links = re.findall(r'<audio[^>]*src=[\'"]([^\'"]+)[\'"]', html)
    if not audio_links:
        audio_links = re.findall(r'<source[^>]*src=[\'"]([^\'"]+)[\'"]', html)
    if not audio_links:
        audio_links = re.findall(r'href=[\'"](/images/[^\'"]+\.(?:mp3|wav|ogg))[\'"]', html)
        
    if not audio_links:
        print(f"  [!] No audio link found on {url}")
        return {}

    audio_url = audio_links[0]
    if audio_url.startswith("/"):
        audio_url = "https://www.sigidwiki.com" + audio_url
        
    ext = os.path.splitext(audio_url.split('?')[0])[1] or ".mp3"
    local_filename = f"{name}{ext}"
    local_path = os.path.join(output_dir, local_filename)
    
    if not os.path.exists(local_path):
        print(f"  [+] Downloading audio: {audio_url} -> {local_path}...")
        audio_req = urllib.request.Request(audio_url, headers=HEADERS)
        with urllib.request.urlopen(audio_req, timeout=20) as a_resp:
            with open(local_path, "wb") as f_out:
                f_out.write(a_resp.read())
        print(f"  [+] Downloaded: {os.path.getsize(local_path):,} bytes")
    else:
        print(f"  [✓] File already present: {local_path}")
        
    # Convert MP3 to WAV using ffmpeg if WAV does not exist
    wav_path = os.path.join(output_dir, f"{name}.wav")
    if not os.path.exists(wav_path):
        cmd = ["ffmpeg", "-y", "-i", local_path, "-ar", "44100", wav_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  [+] Converted to WAV: {wav_path} ({os.path.getsize(wav_path):,} bytes)")
        else:
            print(f"  [!] Conversion failed: {res.stderr[:200]}")
    
    # Extract metadata from table if present
    bandwidth_m = re.search(r'<th>\s*Bandwidth\s*</th>\s*<td>\s*(.*?)\s*</td>', html, re.DOTALL | re.IGNORECASE)
    bw_str = bandwidth_m.group(1).strip() if bandwidth_m else "N/A"
    bw_str = re.sub(r'<[^>]+>', '', bw_str)
    
    mode_m = re.search(r'<th>\s*Mode\s*</th>\s*<td>\s*(.*?)\s*</td>', html, re.DOTALL | re.IGNORECASE)
    mode_str = mode_m.group(1).strip() if mode_m else "N/A"
    mode_str = re.sub(r'<[^>]+>', '', mode_str)

    mod_m = re.search(r'<th>\s*Modulation\s*</th>\s*<td>\s*(.*?)\s*</td>', html, re.DOTALL | re.IGNORECASE)
    mod_str = mod_m.group(1).strip() if mod_m else "N/A"
    mod_str = re.sub(r'<[^>]+>', '', mod_str)
    
    result = {
        "name": name,
        "url": url,
        "audio_file": wav_path if os.path.exists(wav_path) else local_path,
        "format": "wav" if os.path.exists(wav_path) else ext,
        "sigid_mode": mode_str,
        "sigid_modulation": mod_str,
        "sigid_bandwidth": bw_str,
        "expected": page_info
    }
    return result


def main():
    output_dir = os.path.join("C:\\Users\\Yogi\\ntro_signal_analyzer", "verified_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    catalog = []
    for page in TARGET_PAGES:
        try:
            res = fetch_page_audio_and_meta(page, output_dir)
            if res:
                catalog.append(res)
        except Exception as e:
            print(f"  [!] Failed for {page['name']}: {e}")
            
    catalog_path = os.path.join(output_dir, "samples_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    print(f"\n[+] Successfully gathered & converted {len(catalog)} verified signals into {output_dir}")


if __name__ == "__main__":
    main()
