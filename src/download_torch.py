import os
import requests
import time

url = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-win_amd64.whl"
dest = "torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"

def download_file(url, dest):
    existing_size = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"Resuming download from byte {existing_size}...")
    
    mode = "ab" if existing_size > 0 else "wb"
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=120)
        
        # Check if server supports resume (status code 206)
        if existing_size > 0 and response.status_code != 206:
            print("Server does not support resume or range was out of bounds. Restarting download...")
            existing_size = 0
            mode = "wb"
            response = requests.get(url, stream=True, timeout=120)
            
        content_length = response.headers.get('content-length')
        if content_length:
            total_size = int(content_length) + existing_size
        else:
            total_size = existing_size
            
        print(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        with open(dest, mode) as f:
            downloaded = existing_size
            start_time = time.time()
            last_print = time.time()
            for chunk in response.iter_content(chunk_size=8192*8):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # print progress every 1 second to keep log readable
                    if time.time() - last_print > 1.0 or downloaded == total_size:
                        elapsed = time.time() - start_time
                        speed = (downloaded - existing_size) / (1024*1024*elapsed) if elapsed > 0 else 0
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        print(f"Progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB) | Speed: {speed:.2f} MB/s", flush=True)
                        last_print = time.time()
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Error during download chunking: {str(e)}")
        return False

# Retry loop
max_retries = 50
for i in range(max_retries):
    print(f"\nAttempt {i+1}/{max_retries}...")
    success = download_file(url, dest)
    if success:
        break
    print("Waiting 10 seconds before retrying...")
    time.sleep(10)

