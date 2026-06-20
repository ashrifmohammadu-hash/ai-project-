# Frontend Leaf Detection Testing Guide

## Prerequisites

1. Install TensorFlow.js dependencies:
   ```bash
   cd web
   npm install @tensorflow/tfjs @tensorflow-models/mobilenet
   ```

2. Start the frontend dev server:
   ```bash
   cd web
   npm run dev
   ```

3. Start the Flask backend:
   ```bash
   cd plant-disease-backend
   python app.py
   ```

## Manual Test Cases

### Test 1: Non-leaf image rejection
1. Open the app in your browser (http://localhost:5173)
2. Upload a photo of:
   - A **human face / person**
   - A **car**
   - A **building**
   - A **pet (cat/dog)**
   - A **food item** (pizza, burger, etc.)
   - A **solid color image**
3. **Expected result:** The app shows the error message:
   > "Unidentified image. Please upload a clear photo of a plant leaf."
4. The "Get Diagnosis" button should remain **disabled**.
5. The image should **NOT** be sent to the backend.

### Test 2: Leaf image accepted
1. Upload a photo of a **real plant leaf** (from any plant).
2. **Expected result:** The leaf check message appears briefly ("Checking if the image is a leaf..."), then disappears.
3. The "What could be the reason?" section should appear.
4. The "Get Diagnosis" button should become **enabled**.
5. Clicking it sends the image to the backend.

### Test 3: Blurry / low-quality leaf
1. Upload a very blurry or dark photo that contains a leaf but is not clear.
2. **Expected result:** The leaf detection may pass (MobileNet recognizes it as plant-related), but the backend should return low confidence (<30%) and show the "Could not identify" message.

### Test 4: Confidence threshold
1. Upload a leaf photo that is mostly green but very ambiguous (e.g., just the tip of a leaf).
2. The frontend leaf detection should pass if MobileNet is confident it's plant-related.
3. The backend should return either a low-confidence prediction or "unidentified".

### Test 5: Camera capture – take a photo
1. Click the **"📷 Take Photo"** button.
2. **Expected result:** A camera overlay opens showing live video from your device camera.
3. Point the camera at a **plant leaf** and click the **"📷 Capture"** button.
4. **Expected result:** A still image preview appears with **"Retake"** and **"Use Photo"** buttons.
5. Click **"Use Photo"**.
6. **Expected result:** The camera closes. The captured photo appears as the uploaded image preview. Leaf detection runs automatically.

### Test 6: Camera capture – non-leaf rejection
1. Click **"📷 Take Photo"** and point the camera at a **person, book, or object** (not a leaf).
2. Click **"📷 Capture"** → **"Use Photo"**.
3. **Expected result:** The leaf detection should reject it: *"The uploaded image does not appear to be a leaf..."*

### Test 7: Camera capture – retake
1. Click **"📷 Take Photo"**, capture a photo.
2. Click **"Retake"**.
3. **Expected result:** The live camera view returns. You can take another photo.

### Test 8: Camera fallback (no camera / denied)
1. If on a device without a camera (or deny permissions):
2. Click **"📷 Take Photo"**.
3. **Expected result:** A friendly error message appears: *"No camera found"/"Camera access denied"* with a **"Back to file upload"** button.

## Expected Behavior Summary

| Image / Input Type | Frontend (TF.js) | Backend (Groq + CV) | Final Display |
|---|---|---|---|---|
| Person / Animal | ❌ Rejected | Never reached | "Unidentified image" |
| Car / Building | ❌ Rejected | Never reached | "Unidentified image" |
| Food / Object | ❌ Rejected | Never reached | "Unidentified image" |
| Solid color | ❌ Rejected | Never reached | "Unidentified image" |
| Clear leaf photo | ✅ Passed | ✅ Diagnosed | Disease info + explanation |
| Camera capture (leaf) | ✅ Passed | ✅ Diagnosed | Disease info + explanation |
| Camera capture (non-leaf) | ❌ Rejected | Never reached | "Unidentified image" |
| Camera denied / no camera | ❌ N/A | N/A | Camera error + fallback to upload |
| Blurry leaf | ✅ May pass | ⚠️ Low conf / rejected | Disease or "unidentified" |
| Non-plant green object (e.g., green plastic) | ⚠️ May pass | ❌ CV shape rejects | "Unidentified image" |

## Debugging

If a non-leaf image passes the frontend check:
- Open browser DevTools → Console
- Look for `[leafDetector]` log messages
- Check the MobileNet top-3 predictions and their probabilities
- Adjust the `LEAF_KEYWORDS` or threshold in `leafDetector.js` if needed

If a leaf image is incorrectly rejected:
- Check MobileNet predictions in console
- The model may not have this plant in its training set
- Consider adding more keywords to `LEAF_KEYWORDS` in `leafDetector.js`
