import base64
import json

# Read the JSON file
with open('foodorderapp.json', 'r') as f:
    json_content = f.read()

# Encode to base64
encoded = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')

print("✅ Base64 encoded credentials:")
print(encoded)
print(f"\nLength: {len(encoded)}")
print("\nCopy this to your .env file as:")
print("GOOGLE_APPLICATION_CREDENTIALS=" + encoded)
