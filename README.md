```
sudo apt install ffmpeg

virtualenv venv
. venv/bin/activate
pip install -r requirements.txt

python sender.py
```

Open VLC
In VLC, do media -> open network stream -> set url to uudp://@localhost:2222?pkt_size=1316 as seen in terminal.