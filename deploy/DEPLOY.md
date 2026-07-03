# Deploying the MACT app on a Yotta VM behind nginx

The standalone entry point is `mact_app.py` (it shares all logic with the
combined app via `arbitration_studio/mact_ui.py`). These steps put it behind
nginx as a systemd service.

## 1. System packages
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip nginx poppler-utils
# poppler-utils is REQUIRED for the Gemini OCR (pdf2image rasterization).
```

## 2. Get the code
```bash
sudo git clone https://github.com/parth713/Arbitration-Studio.git /opt/arbitration-studio
cd /opt/arbitration-studio
sudo python3 -m venv .venv
sudo .venv/bin/pip install -r requirements.txt
```

## 3. Secrets — create `/opt/arbitration-studio/.env`
The app reads this via python-dotenv (note the **leading dot**: `.env`, not `env`).
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GOOGLE_API_KEY=...            # Vertex AI Express key for Gemini OCR
OCR_MODEL=gemini-3-pro-preview
OCR_DPI=150                   # 150 keeps memory low; raise to 300 for max OCR quality
```
Lock it down: `sudo chmod 600 .env && sudo chown www-data:www-data .env`

## 4. Run as a service
```bash
sudo cp deploy/mact.service /etc/systemd/system/mact.service
sudo chown -R www-data:www-data /opt/arbitration-studio
sudo systemctl daemon-reload
sudo systemctl enable --now mact
sudo systemctl status mact          # should be "active (running)"
curl -sI http://127.0.0.1:8501 | head -1   # expect HTTP 200
```
`.streamlit/config.toml` in the repo binds Streamlit to 127.0.0.1:8501 and sets
the 50 MB upload limit — no extra flags needed.

## 5. nginx reverse proxy
```bash
sudo cp deploy/nginx-mact.conf /etc/nginx/sites-available/mact
# edit server_name to your domain / public IP
sudo ln -s /etc/nginx/sites-available/mact /etc/nginx/sites-enabled/mact
sudo nginx -t && sudo systemctl reload nginx
```
Open `http://<your-server>/`. The **WebSocket headers** in the nginx config are
essential — without them the page loads but never connects ("Please wait…").

## 6. HTTPS (recommended)
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d mact.example.com
```

## Updating after a push
```bash
cd /opt/arbitration-studio
sudo git pull
sudo .venv/bin/pip install -r requirements.txt   # only if deps changed
sudo systemctl restart mact
```

## Notes
- **Memory:** OCR rasterization is the heavy part. `OCR_DPI=150` plus the
  `MemoryMax=3G` in the service file keep it stable; give the VM ≥ 2 GB RAM.
- **One worker:** Streamlit is single-process. For several concurrent judges,
  run more instances on different ports (8501, 8502, …) and load-balance them
  in nginx with an `upstream` block.
- **Combined app instead:** to serve the arbitration+MACT app, point
  `ExecStart` at `app.py` instead of `mact_app.py`.
