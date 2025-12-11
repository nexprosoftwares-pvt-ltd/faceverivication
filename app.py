from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import base64
import cv2
import re

app = Flask(__name__)

# Convert base64 → OpenCV Image
def decode_image(base64_string):
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: cv2.imdecode returned None")
            return None
            
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# ========== ENCODE FACE ==========
@app.route("/encode", methods=["POST"])
def encode_face():
    try:
        data = request.json
        image_base64 = data.get("image")

        if not image_base64:
            return jsonify({"success": False, "error": "No image provided"}), 400

        print("Received image data, attempting to decode...")
        img = decode_image(image_base64)
        
        if img is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        print(f"Image decoded successfully. Shape: {img.shape}")

        # Convert BGR → RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print("Detecting faces...")
        encodings = face_recognition.face_encodings(rgb)

        if len(encodings) == 0:
            return jsonify({"success": False, "error": "No face detected in image. Please ensure your face is clearly visible."}), 400

        print(f"Face detected! Generated {len(encodings[0])} dimension embedding")
        
        return jsonify({
            "success": True,
            "embedding": encodings[0].tolist()
        })
    
    except Exception as e:
        print(f"Error in encode_face: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ========== COMPARE FACES ==========
@app.route("/compare", methods=["POST"])
def compare_faces():
    try:
        data = request.json
        
        if not data.get("embedding1") or not data.get("embedding2"):
            return jsonify({
                "success": False,
                "error": "Missing embeddings"
            }), 400

        try:
            import json
            
            # Handle both string and array inputs
            emb1_raw = data.get("embedding1")
            emb2_raw = data.get("embedding2")
            
            # If string, parse as JSON
            if isinstance(emb1_raw, str):
                emb1_raw = json.loads(emb1_raw)
            if isinstance(emb2_raw, str):
                emb2_raw = json.loads(emb2_raw)
            
            # Convert to numpy arrays
            emb1 = np.array(emb1_raw, dtype=float)
            emb2 = np.array(emb2_raw, dtype=float)
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": "Invalid embedding format",
                "details": str(e)
            }), 400

        # Calculate distance
        distance = np.linalg.norm(emb1 - emb2)
        match = bool(distance < 0.6)

        return jsonify({
            "success": True,
            "distance": float(distance),
            "match": match
        }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        
        
        
# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "face-recognition"}), 200

# Run server
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Face Recognition Service...")
    print("Server running on http://127.0.0.1:8080")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8080, debug=True)