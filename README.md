```
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt

python sender.py
```

Open VLC
In VLC, do media -> open network stream -> set url to udp://@localhost:2222 as seen in terminal.