"""Google Colab setup for ChronoDoc.

In Colab: Runtime > Change runtime type > T4 GPU, then run this file's cells
or copy the commands into separate notebook cells. Colab GPU sessions are free
but temporary and can disconnect when idle.
"""

# Cell 1: install the GPU runtime and project dependencies.
# In a Colab cell, run:
# !pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# !pip install -q streamlit transformers diffusers datasets accelerate safetensors soundfile sentencepiece

# Cell 2: verify the GPU.
# import torch
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")
# assert torch.cuda.is_available(), "Select a GPU runtime before continuing"

# Cell 3: upload this Historical Cronodoc folder to /content, or clone its repository.
# from google.colab import files
# uploaded = files.upload()
# !unzip -q Historical_Cronodoc.zip -d /content
# %cd /content/Historical Cronodoc

# Cell 4: launch Streamlit through a temporary public tunnel.
# !pip install -q localtunnel
# import subprocess
# streamlit = subprocess.Popen(["streamlit", "run", "crono_app.py", "--server.headless", "true", "--server.port", "8501"])
# !npx localtunnel --port 8501
