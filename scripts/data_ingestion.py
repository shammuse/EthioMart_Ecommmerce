from telethon.sync import TelegramClient
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_ID = os.getenv('API_ID')        # Fetch API_ID from .env
API_HASH = os.getenv('API_HASH')    # Fetch API_HASH from .env
CHANNELS = ['@aradabrand2']         # Add other channels here

OUTPUT_DIR = "../data/raw/"         # Output directory for raw data

def scrape_telegram():
    """
    Scrapes messages from specified Telegram channels and saves them as JSON files.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists
    with TelegramClient('session_name', API_ID, API_HASH) as client:
        for channel in CHANNELS:
            messages = []
            print(f"Fetching messages from {channel}...")
            for message in client.iter_messages(channel):
                messages.append({
                    'id': message.id,
                    'text': message.text,
                    'date': message.date.isoformat(),
                    'sender': message.sender_id,
                })
            # Save messages to a JSON file
            output_file = os.path.join(OUTPUT_DIR, f"{channel.strip('@')}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(messages, f, ensure_ascii=False, indent=4)
            print(f"Saved messages from {channel} to {output_file}")

if __name__ == "__main__":
    scrape_telegram()

