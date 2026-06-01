# Stock Portfolio Optimization Dashboard

This is a deploy-ready Dash app converted from the original Jupyter Notebook.

## Files

- `portfolio_dashboard_app.py` . Main Dash application
- `requirements_portfolio_dashboard.txt` . Python dependencies

## Local Run

```bash
pip install -r requirements_portfolio_dashboard.txt
python portfolio_dashboard_app.py
```

Open:

```text
http://localhost:7860
```

## Hugging Face Spaces Deployment

1. Create a new Space.
2. Select SDK: Docker or Blank.
3. Upload the app file as `app.py`.
4. Upload the requirements file as `requirements.txt`.
5. The app runs on port `7860`.

## Render Deployment

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:server --bind 0.0.0.0:$PORT
```

## Notes

- The dashboard fetches live stock data using Yahoo Finance via `yfinance`.
- Internet access is required in the deployment environment.
- This is for educational use only and is not financial advice.
