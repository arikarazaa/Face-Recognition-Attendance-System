import os
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import pytz

_app = None


def firebase_init(service_account_path='serviceAccountKey.json', db_url=None):
    """
    Initialize Firebase Admin SDK (Realtime Database).
    Call once at startup before any log_attendance() calls.

    Args:
        service_account_path: Path to your Firebase service account JSON file.
        db_url: Your Realtime Database URL, e.g.
                https://your-project-id.firebaseio.com
    """
    global _app
    if _app is not None:
        return _app  # Already initialized

    if not os.path.exists(service_account_path):
        raise FileNotFoundError(
            f"Service account file not found: '{service_account_path}'\n"
            "Download it from Firebase Console → Project Settings → Service Accounts."
        )
    if db_url is None:
        raise ValueError("You must provide your Firebase Realtime Database URL.")

    cred = credentials.Certificate(service_account_path)
    _app = firebase_admin.initialize_app(cred, {'databaseURL': db_url})
    return _app


def log_attendance(name: str, confidence: float, tz: str = 'Asia/Karachi'):
    """
    Write an attendance entry under:
        attendance/{YYYY-MM-DD}/{name}/{push_id}
              → { timestamp, confidence }

    Args:
        name       : Recognized person's name.
        confidence : Model confidence score (0.0 – 1.0).
        tz         : Timezone string for timestamp/date grouping.
    """
    tzinfo    = pytz.timezone(tz)
    now       = datetime.now(tzinfo)
    today     = now.strftime('%Y-%m-%d')
    timestamp = now.isoformat()

    ref = db.reference(f'attendance/{today}/{name}')
    ref.push({'timestamp': timestamp, 'confidence': float(confidence)})
    print(f"  ✅ Logged: {name} | conf={confidence:.2f} | {timestamp}")