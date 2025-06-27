import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os

# === Load environment variables
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
CSV_PATH = "predictions/tft_forecast_enriched.csv"

# === Debug print
print("📄 Debug Mode: Email Configuration")
print("SENDER_EMAIL     :", SENDER_EMAIL)
print("RECEIVER_EMAIL   :", RECEIVER_EMAIL)
print("APP_PASSWORD (first 4 chars):", APP_PASSWORD[:4], "..." if APP_PASSWORD else "None")

# === Check file
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File not found: {CSV_PATH}")

# === Compose email
msg = EmailMessage()
msg["Subject"] = "🔮 TFT Forecast Report"
msg["From"] = SENDER_EMAIL
msg["To"] = RECEIVER_EMAIL
msg.set_content("Hi,\n\nAttached is the latest 7-day forecast report enriched with metadata.\n\n— Automated Reporting Bot")

# === Attach CSV
with open(CSV_PATH, "rb") as f:
    msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=os.path.basename(CSV_PATH))

# === Try sending
try:
    print("📤 Connecting to smtp.gmail.com:465...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        print("🔐 Logging in...")
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)
    print("✅ Email sent successfully!")

except smtplib.SMTPAuthenticationError as e:
    print("❌ SMTP Authentication Error (likely bad credentials or app password)")
    print("Code:", e.smtp_code)
    print("Message:", e.smtp_error.decode())

except Exception as e:
    print("❌ General email send failure:", e)
