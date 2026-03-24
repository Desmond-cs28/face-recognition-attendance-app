import face_recognition
import os
import pickle

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "encodings.pkl"

known_encodings = []
known_names     = []

print("[INFO] Encoding faces...\n")




files = [f for f in os.listdir(KNOWN_FACES_DIR)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not files:
    print(f"⚠️  No photos found in '{KNOWN_FACES_DIR}' folder.")
    print(f"    → Add student photos named like: Desmond_Zambogunaa.jpg")
    exit()

for filename in files:
    
    name = os.path.splitext(filename)[0].replace("_", " ")
    path = os.path.join(KNOWN_FACES_DIR, filename)

    print(f"  Processing: {name} ({filename})")

    image     = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"  ⚠️  No face detected in {filename} — skipping.")
        print(f"      Retake photo: good lighting, face clearly visible.\n")
        continue

    if len(encodings) > 1:
        print(f"  ⚠️  Multiple faces in {filename} — using first face only.\n")

    known_encodings.append(encodings[0])
    known_names.append(name)
    print(f"  ✅  Encoded successfully: {name}\n")

if not known_encodings:
    print("❌  No faces were encoded. Check your photos and try again.")
    exit()

# Save encodings to file
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("=" * 50)
print(f"✅  Done! Encoded {len(known_names)} student(s).")
print(f"    Names: {known_names}")
print(f"    Saved to: {ENCODINGS_FILE}")
print("=" * 50)


