# predict.py – Pipeline: OpenCV gate → Histogram check → Gemini (main) → MobileNetV3 (fallback) → RF (metadata)
import cv2
import numpy as np
import os
from pathlib import Path
import logging

# ── Logging helper ──
def _log(step, message):
    try:
        from flask import current_app
        current_app.logger.info(f"[{step}] {message}")
    except Exception:
        try:
            print(f"[{step}] {message}")
        except UnicodeEncodeError:
            safe = message.encode("ascii", "replace").decode("ascii")
            print(f"[{step}] {safe}")

# Lazy-loaded PyTorch model globals
_PYTORCH_MODEL = None
_PYTORCH_CLASSES = None
_PYTORCH_DEVICE = None

# ---------------------------------------------------------------------------
# FULL DISEASE_INFO – all 70+ diseases from the PlantVillage + Sri Lanka dataset
# ---------------------------------------------------------------------------
DISEASE_INFO = {
    "Apple_Cedar_apple_rust": {
        "cause": "Fungus (Gymnosporangium juniperi-virginianae)",
        "weather": "Wet spring weather, rain and high humidity",
        "insects": "Not caused by insects — spreads between cedar and apple trees",
        "treatment": "Apply myclobutanil or mancozeb fungicide in early spring",
        "symptoms": "Orange rust spots on leaves and fruit; cedar-apple rust lifecycle",
        "prevention": "Remove nearby cedar trees if possible; apply preventive fungicide",
        "fertilizer": "Standard apple orchard NPK",
        "severity": "Medium",
    },
    "Apple___Apple_scab": {
        "cause": "Fungus (Venturia inaequalis)",
        "weather": "Cool wet spring weather, rain and humidity",
        "insects": "Not caused by insects",
        "treatment": "Apply myclobutanil fungicide, prune infected branches, remove fallen leaves",
        "symptoms": "Olive-green velvety spots on leaves and fruit",
        "prevention": "Remove fallen leaves; apply preventive fungicide at green tip",
        "fertilizer": "Standard apple orchard NPK",
        "severity": "High",
    },
    "Apple___Black_rot": {
        "cause": "Fungus (Botryosphaeria obtusa)",
        "weather": "Warm humid weather, temperatures 20-25°C",
        "insects": "Insects can spread through wounds on fruit",
        "treatment": "Remove infected fruit and branches, apply captan fungicide",
        "symptoms": "Purple leaf spots; black rotted fruit",
        "prevention": "Prune dead wood; remove fruit mummies",
        "fertilizer": "Standard apple orchard NPK",
        "severity": "High",
    },
    "Apple_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green apple leaves",
        "prevention": "Standard orchard management",
        "fertilizer": "Standard apple orchard NPK",
        "severity": "Low",
    },
    "Blueberry_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green blueberry leaves",
        "prevention": "Standard blueberry management",
        "fertilizer": "Acid-loving plant fertilizer",
        "severity": "Low",
    },
    "Cherry_(including_sour)_Powdery_mildew": {
        "cause": "Fungus (Podosphaera clandestina)",
        "weather": "Warm dry weather with high humidity at night",
        "insects": "Not caused by insects",
        "treatment": "Apply sulfur or potassium bicarbonate fungicide, improve air circulation",
        "symptoms": "White powdery coating on leaves",
        "prevention": "Prune for airflow; avoid overhead watering",
        "fertilizer": "Standard cherry orchard NPK",
        "severity": "Medium",
    },
    "Cherry_(including_sour)_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green cherry leaves",
        "prevention": "Standard orchard management",
        "fertilizer": "Standard cherry orchard NPK",
        "severity": "Low",
    },
    "Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot": {
        "cause": "Fungus (Cercospora zeae-maydis)",
        "weather": "High humidity, warm temperatures 25-30°C, heavy dew",
        "insects": "Not caused by insects",
        "symptoms": "Rectangular gray-tan lesions between veins on long narrow corn leaves",
        "treatment": "Apply strobilurin fungicide, use resistant varieties, crop rotation",
        "prevention": "Resistant hybrids, field sanitation, avoid dense planting",
        "fertilizer": "Balanced NPK for maize per local extension guidelines",
        "severity": "Medium",
    },
    "Corn_(maize)_Common_rust_": {
        "cause": "Fungus (Puccinia sorghi)",
        "weather": "Cool humid weather, temperatures 15-20°C, heavy dew",
        "insects": "Not caused by insects — spreads by wind",
        "treatment": "Apply propiconazole fungicide, plant resistant hybrid varieties",
        "symptoms": "Reddish-brown pustules on both leaf surfaces",
        "prevention": "Resistant hybrids; early planting",
        "fertilizer": "Balanced NPK for maize",
        "severity": "Medium",
    },
    "Corn_(maize)_Northern_Leaf_Blight": {
        "cause": "Fungus (Exserohilum turcicum)",
        "weather": "Moderate temperatures, high humidity, frequent rain",
        "insects": "Not caused by insects",
        "treatment": "Apply azoxystrobin fungicide, use resistant varieties, crop rotation",
        "symptoms": "Long elliptical gray-green lesions on lower leaves",
        "prevention": "Resistant hybrids; crop rotation; field sanitation",
        "fertilizer": "Balanced NPK for maize",
        "severity": "High",
    },
    "Corn_(maize)_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green corn leaves",
        "prevention": "Standard corn management",
        "fertilizer": "Balanced NPK for maize",
        "severity": "Low",
    },
    "Grape_Black_rot": {
        "cause": "Fungus (Guignardia bidwellii)",
        "weather": "Warm wet weather, 24-30°C, frequent rain and leaf wetness",
        "insects": "Not caused by insects — spreads by rain splash and wind",
        "symptoms": "Circular tan to brown spots with dark reddish-brown borders; shriveled black fruit mummies",
        "treatment": "Remove infected leaves and fruit mummies. Apply mancozeb or myclobutanil fungicide every 7-14 days.",
        "prevention": "Prune for airflow, avoid overhead irrigation, sanitize pruners",
        "fertilizer": "Balanced NPK; avoid excess nitrogen",
        "severity": "High",
    },
    "Grape_Esca_(Black_Measles)": {
        "cause": "Fungal complex (Phaeomoniella, Phaeoacremonium)",
        "weather": "Hot dry summers after wet winters",
        "insects": "Not caused by insects",
        "symptoms": "Tiger-stripe wood, leaf scorch, white rot in trunk; gradual vine decline",
        "treatment": "No cure — remove infected vines, avoid large pruning wounds",
        "prevention": "Avoid winter pruning in rain, train vines on good drainage",
        "fertilizer": "Maintain moderate vigor; avoid stress from drought or over-fertilization",
        "severity": "High",
    },
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cause": "Fungus (Isariopsis clavispora)",
        "weather": "High humidity and warm temperatures",
        "insects": "Not caused by insects",
        "symptoms": "Angular to irregular brown leaf spots, often near veins",
        "treatment": "Apply copper-based fungicide, improve vineyard air circulation",
        "prevention": "Leaf removal at dormancy, canopy management",
        "fertilizer": "Standard vineyard NPK based on soil test",
        "severity": "Medium",
    },
    "Grape_healthy": {
        "cause": "No disease detected",
        "weather": "Normal growing conditions for grapevine",
        "insects": "No significant pest damage detected",
        "symptoms": "Green lobed grape leaves without characteristic disease lesions",
        "treatment": "No treatment needed — continue regular vineyard care",
        "prevention": "Monitor regularly, maintain pruning and airflow",
        "fertilizer": "Apply nutrients per soil test",
        "severity": "Low",
    },
    "Orange_Haunglongbing_(Citrus_greening)": {
        "cause": "Bacteria (Candidatus Liberibacter asiaticus)",
        "weather": "Spreads in all weather conditions",
        "insects": "Asian citrus psyllid insect spreads this disease",
        "treatment": "No cure — remove infected trees, control psyllid insects with insecticide",
        "symptoms": "Yellow mottled leaves, blotchy green veins",
        "prevention": "Control psyllid populations; use certified disease-free nursery stock",
        "fertilizer": "Standard citrus NPK",
        "severity": "High",
    },
    "Peach_Bacterial_spot": {
        "cause": "Bacteria (Xanthomonas campestris)",
        "weather": "Warm wet weather, rain and wind spread bacteria",
        "insects": "Not caused by insects",
        "treatment": "Apply copper-based bactericide, avoid overhead irrigation",
        "symptoms": "Small dark angular spots on leaves, fruit lesions",
        "prevention": "Resistant varieties; avoid working wet trees",
        "fertilizer": "Standard peach orchard NPK",
        "severity": "Medium",
    },
    "Peach_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green peach leaves",
        "prevention": "Standard orchard management",
        "fertilizer": "Standard peach orchard NPK",
        "severity": "Low",
    },
    "Pepper,_bell_Bacterial_spot": {
        "cause": "Bacteria (Xanthomonas campestris pv. vesicatoria)",
        "weather": "Warm wet weather, temperatures 25-30°C, heavy rain",
        "insects": "Not caused by insects — spreads by water splash",
        "treatment": "Apply copper bactericide, use disease-free seeds, avoid overhead watering",
        "symptoms": "Water-soaked spots on leaves and fruit",
        "prevention": "Disease-free seed; crop rotation; avoid working wet plants",
        "fertilizer": "Balanced NPK for peppers",
        "severity": "Medium",
    },
    "Pepper,_bell_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green bell pepper leaves",
        "prevention": "Standard pepper management",
        "fertilizer": "Balanced NPK for peppers",
        "severity": "Low",
    },
    "Potato_Early_blight": {
        "cause": "Fungus (Alternaria solani)",
        "weather": "Warm humid weather, temperatures 24-29°C, heavy dew",
        "insects": "Not caused by insects",
        "treatment": "Apply mancozeb or chlorothalonil fungicide, remove infected leaves",
        "symptoms": "Dark brown spots with concentric rings — target-like pattern",
        "prevention": "Crop rotation; avoid overhead irrigation; resistant varieties",
        "fertilizer": "Balanced NPK; avoid excess nitrogen",
        "severity": "Medium",
    },
    "Potato_Late_blight": {
        "cause": "Fungus (Phytophthora infestans)",
        "weather": "Cool moist conditions, temperatures 10-20°C, high rainfall",
        "insects": "Not caused by insects — spreads by wind and rain",
        "treatment": "Apply metalaxyl fungicide, destroy infected plants immediately",
        "symptoms": "Water-soaked lesions, white mold on leaf undersides",
        "prevention": "Resistant varieties; destroy volunteer potatoes; good drainage",
        "fertilizer": "Balanced potato NPK",
        "severity": "High",
    },
    "Potato_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green potato leaves",
        "prevention": "Standard potato management",
        "fertilizer": "Balanced potato NPK",
        "severity": "Low",
    },
    "Raspberry_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green raspberry leaves",
        "prevention": "Standard raspberry management",
        "fertilizer": "Balanced berry NPK",
        "severity": "Low",
    },
    "Soybean_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green soybean leaves",
        "prevention": "Standard soybean management",
        "fertilizer": "Soybean inoculant; balanced NPK",
        "severity": "Low",
    },
    "Squash_Powdery_mildew": {
        "cause": "Fungus (Podosphaera xanthii)",
        "weather": "Warm dry weather, temperatures 20-25°C, high humidity at night",
        "insects": "Not caused by insects",
        "treatment": "Apply sulfur or neem oil fungicide, improve air circulation",
        "symptoms": "White powdery coating on upper leaf surfaces",
        "prevention": "Resistant varieties; adequate spacing; avoid overhead watering",
        "fertilizer": "Balanced NPK for cucurbits",
        "severity": "Medium",
    },
    "Strawberry_Leaf_scorch": {
        "cause": "Fungus (Diplocarpon earlianum)",
        "weather": "Warm wet weather, high humidity and rain",
        "insects": "Not caused by insects",
        "treatment": "Apply captan fungicide, remove infected leaves, avoid overhead irrigation",
        "symptoms": "Purple-brown spots that merge, giving scorched appearance",
        "prevention": "Remove old leaves after harvest; avoid overhead watering",
        "fertilizer": "Balanced strawberry NPK",
        "severity": "Medium",
    },
    "Strawberry_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green strawberry leaves",
        "prevention": "Standard strawberry management",
        "fertilizer": "Balanced strawberry NPK",
        "severity": "Low",
    },
    "Tomato_Bacterial_spot": {
        "cause": "Bacteria (Xanthomonas campestris)",
        "weather": "Warm wet weather, temperatures 25-30°C, rain and wind",
        "insects": "Not caused by insects — spreads by water splash",
        "treatment": "Apply copper bactericide, use resistant varieties, avoid overhead watering",
        "symptoms": "Small dark greasy spots with yellow halos on leaves and fruit",
        "prevention": "Disease-free seed; crop rotation; copper sprays preventively",
        "fertilizer": "Balanced tomato NPK",
        "severity": "Medium",
    },
    "Tomato_Early_blight": {
        "cause": "Fungus (Alternaria solani)",
        "weather": "Warm humid weather, heavy dew, temperatures 24-29°C",
        "insects": "Not caused by insects",
        "treatment": "Apply mancozeb fungicide, crop rotation, remove lower infected leaves",
        "symptoms": "Dark brown spots with concentric rings on lower leaves first",
        "prevention": "Mulch; avoid overhead watering; crop rotation",
        "fertilizer": "Balanced NPK; avoid excess nitrogen",
        "severity": "Medium",
    },
    "Tomato_Late_blight": {
        "cause": "Fungus (Phytophthora infestans)",
        "weather": "Cool wet weather, humidity above 90%, temperatures 10-20°C",
        "insects": "Not caused by insects — spreads by wind and rain",
        "treatment": "Apply copper fungicide, remove infected leaves immediately, avoid wetting leaves",
        "symptoms": "Water-soaked dark lesions on leaves and stems; white mold in high humidity",
        "prevention": "Resistant varieties; destroy volunteer tomatoes; good air circulation",
        "fertilizer": "Balanced tomato NPK",
        "severity": "High",
    },
    "Tomato_Leaf_Mold": {
        "cause": "Fungus (Passalora fulva)",
        "weather": "High humidity above 85%, poor air circulation, temperatures 20-25°C",
        "insects": "Not caused by insects",
        "treatment": "Apply chlorothalonil fungicide, improve ventilation, reduce humidity",
        "symptoms": "Yellow spots on upper surface; olive-green mold on lower leaf surface",
        "prevention": "Reduce humidity; improve air circulation; resistant varieties",
        "fertilizer": "Balanced tomato NPK",
        "severity": "Medium",
    },
    "Tomato_Septoria_leaf_spot": {
        "cause": "Fungus (Septoria lycopersici)",
        "weather": "Warm wet weather, temperatures 20-25°C, rain and overhead watering",
        "insects": "Not caused by insects",
        "treatment": "Apply mancozeb fungicide, remove infected leaves, avoid wetting foliage",
        "symptoms": "Small circular spots with dark borders and gray centers",
        "prevention": "Crop rotation; remove lower leaves; avoid overhead watering",
        "fertilizer": "Balanced tomato NPK",
        "severity": "Medium",
    },
    "Tomato_Spider_mites_Two-spotted_spider_mite": {
        "cause": "Spider mite pest (Tetranychus urticae)",
        "weather": "Hot dry weather, temperatures above 30°C, low humidity",
        "insects": "Caused by spider mites — tiny insects under leaves",
        "treatment": "Apply abamectin or neem oil miticide, increase humidity, remove heavily infested leaves",
        "symptoms": "Fine stippling, webbing on leaf undersides, yellowing",
        "prevention": "Keep plants well-watered; introduce predatory mites",
        "fertilizer": "Balanced tomato NPK",
        "severity": "Medium",
    },
    "Tomato_Target_Spot": {
        "cause": "Fungus (Corynespora cassiicola)",
        "weather": "Warm humid weather, temperatures 25-30°C, high humidity",
        "insects": "Not caused by insects",
        "treatment": "Apply azoxystrobin fungicide, improve air circulation, avoid overhead irrigation",
        "symptoms": "Brown target-like spots with concentric rings",
        "prevention": "Resistant varieties; crop rotation; reduce leaf wetness",
        "fertilizer": "Balanced tomato NPK",
        "severity": "Medium",
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "Virus (Tomato yellow leaf curl virus — TYLCV)",
        "weather": "Spreads in all weather — carried by whitefly insects",
        "insects": "Whitefly (Bemisia tabaci) spreads this virus",
        "treatment": "No cure for virus — remove infected plants, control whitefly with imidacloprid insecticide",
        "symptoms": "Yellow curled leaves, stunted growth",
        "prevention": "Whitefly control; resistant varieties; reflective mulch",
        "fertilizer": "Maintain nutrition; support plant vigour",
        "severity": "High",
    },
    "Tomato_Tomato_mosaic_virus": {
        "cause": "Virus (Tomato mosaic virus — ToMV)",
        "weather": "Spreads in all weather conditions",
        "insects": "Spreads by contact, tools, and aphid insects",
        "treatment": "No cure — remove infected plants, disinfect tools, control aphids",
        "symptoms": "Mottled light and dark green pattern on leaves",
        "prevention": "Resistant varieties; sanitize tools; wash hands",
        "fertilizer": "Maintain balanced nutrition",
        "severity": "Medium",
    },
    "Tomato_healthy": {
        "cause": "No disease detected",
        "weather": "Plant is healthy",
        "insects": "No pest damage detected",
        "treatment": "No treatment needed — continue regular care",
        "symptoms": "Normal green tomato leaves",
        "prevention": "Standard tomato management",
        "fertilizer": "Balanced tomato NPK",
        "severity": "Low",
    },
    "Banana_healthy": {
        "cause": "No disease detected",
        "symptoms": "Green banana leaf with normal colour; no major streaks or wilt",
        "weather": "Typical tropical humidity; maintain good drainage",
        "insects": "Monitor for thrips and nematodes per local practice",
        "treatment": "No treatment needed — balanced NPK and potassium as recommended",
        "prevention": "Remove old leaves, avoid waterlogging, use clean planting material",
        "fertilizer": "Potassium-rich fertilizer per local banana program",
        "severity": "Low",
    },
    "Banana_Sigatoka": {
        "cause": "Fungus (Pseudocercospora fijiensis) — Black/Yellow Sigatoka leaf spot",
        "symptoms": "Dark streaks and spots along the leaf blade, often with yellow halos",
        "weather": "Warm wet humid conditions; spreads in rain and wind",
        "insects": "Not caused by insects",
        "treatment": "Remove heavily infected leaves; improve drainage; apply fungicides (mancozeb, propiconazole)",
        "prevention": "Good spacing, balanced fertilizer, de-leafing, resistant varieties if available",
        "fertilizer": "Maintain potassium; avoid excess nitrogen during heavy infection",
        "severity": "High",
    },
    "Banana_Xanthomonas_wilt": {
        "cause": "Bacteria (Xanthomonas campestris pv. musacearum) — bacterial wilt",
        "symptoms": "Yellowing and wilting of leaves, dark streaks in pseudostem",
        "weather": "Spreads in wet weather; tools and infected planting material spread disease",
        "insects": "Not the primary cause; focus on sanitation and clean suckers",
        "treatment": "No cure for infected mats — rogue and destroy affected plants; use disease-free planting material",
        "prevention": "Use certified suckers, disinfect tools, avoid sharing farm equipment",
        "fertilizer": "Support remaining healthy plants with balanced nutrition after rogueing",
        "severity": "High",
    },
    "Rice_Blast": {
        "cause": "Fungus (Magnaporthe oryzae)",
        "symptoms": "Diamond-shaped lesions with gray centres on leaves",
        "treatment": "Use resistant varieties; tricyclazole or recommended fungicide per DOA; balanced nitrogen",
        "prevention": "Avoid excess nitrogen; field sanitation; resistant varieties",
        "fertilizer": "Split nitrogen applications; follow local rice recommendations",
        "severity": "High",
        "weather": "Humid, cloudy weather with dew",
        "insects": "Not caused by insects",
    },
    "Rice_Bacterial_blight": {
        "cause": "Bacteria (Xanthomonas oryzae pv. oryzae)",
        "symptoms": "Yellowing and drying from leaf tips; wavy margins on lesions",
        "treatment": "Copper-based sprays where approved; use certified seed; rogue infected plants",
        "prevention": "Resistant varieties; avoid overhead irrigation at heading",
        "fertilizer": "Balanced NPK per soil test",
        "severity": "High",
        "weather": "Warm wet seasons",
        "insects": "Not caused by insects",
    },
    "Rice_Brown_spot": {
        "cause": "Fungus (Bipolaris oryzae)",
        "symptoms": "Brown oval spots on leaves; more common on nutrient-stressed plants",
        "treatment": "Mancozeb or propiconazole per extension advice; correct soil fertility",
        "prevention": "Adequate potassium and silicon; clean seed",
        "fertilizer": "Potassium and balanced NPK",
        "severity": "Medium",
        "weather": "Humid conditions",
        "insects": "Not caused by insects",
    },
    "Rice_Tungro": {
        "cause": "Virus complex (spread by green leafhopper)",
        "symptoms": "Stunted plants, yellow-orange leaf discolouration",
        "treatment": "Control leafhopper; rogue infected plants; use resistant varieties",
        "prevention": "Early planting; resistant varieties; hopper monitoring",
        "fertilizer": "Avoid excess nitrogen on infected fields",
        "severity": "High",
        "weather": "Spreads in warm seasons with vector activity",
        "insects": "Green leafhopper spreads the virus",
    },
    "Rice_healthy": {
        "cause": "No disease detected",
        "treatment": "No treatment needed — continue regular care",
        "prevention": "Standard rice management practices",
        "fertilizer": "Balanced NPK per soil test",
        "severity": "Low",
        "weather": "Normal growing conditions",
        "insects": "No pest damage detected",
        "symptoms": "Normal green rice leaves",
    },
    "Mango_healthy": {
        "cause": "No disease detected",
        "treatment": "Continue regular orchard care and nutrition",
        "prevention": "Pruning, sanitation, monitoring",
        "fertilizer": "NPK per age of tree",
        "severity": "Low",
        "weather": "Normal",
        "insects": "Monitor common mango pests",
        "symptoms": "Normal green mango leaves",
    },
    "Mango_Anthracnose": {
        "cause": "Fungus (Colletotrichum gloeosporioides)",
        "symptoms": "Dark spots on leaves and fruit; common in humid mango areas",
        "treatment": "Copper or mancozeb sprays; remove infected twigs; improve airflow",
        "prevention": "Prune canopy; post-harvest care for fruit",
        "fertilizer": "Balanced nutrition; avoid excess nitrogen",
        "severity": "High",
        "weather": "Rainy humid weather",
        "insects": "Not the primary cause",
    },
    "Mango_Bacterial_canker": {
        "cause": "Bacteria (Xanthomonas campestris pv. mangiferaeindicae)",
        "symptoms": "Water-soaked lesions on leaves turning dark; cracking bark",
        "treatment": "Apply copper bactericide; prune infected branches; improve drainage",
        "prevention": "Avoid overhead irrigation; use disease-free nursery stock",
        "fertilizer": "Balanced mango orchard nutrition",
        "severity": "High",
        "weather": "Warm humid weather with rain splash",
        "insects": "Spread by rain splash and wind",
    },
    "Mango_Die_back": {
        "cause": "Fungal complex including Botryosphaeria species",
        "symptoms": "Tip dieback of branches; branch cankers; gum exudation",
        "treatment": "Prune affected branches below visible damage; apply copper fungicide paste to cuts",
        "prevention": "Avoid mechanical injury; maintain tree vigour",
        "fertilizer": "Balanced NPK; avoid water stress",
        "severity": "Medium",
        "weather": "Stress periods, drought or waterlogging",
        "insects": "Secondary pests may enter through dieback areas",
    },
    "Mango_Powdery_mildew": {
        "cause": "Fungus (Oidium mangiferae)",
        "symptoms": "White powdery coating on leaves, flowers and young fruit",
        "treatment": "Apply sulfur or myclobutanil fungicide at flowering stage",
        "prevention": "Prune dense canopy; avoid excess nitrogen",
        "fertilizer": "Balanced NPK",
        "severity": "Medium",
        "weather": "Cool dry weather with high humidity at night",
        "insects": "Not caused by insects",
    },
    "Mango_Sooty_mould": {
        "cause": "Fungus growing on honeydew secreted by insects such as scale, mealybugs, aphids",
        "symptoms": "Black sooty coating on leaf surfaces; may cover fruit",
        "treatment": "Control the insect pests excreting honeydew; apply insecticide for scale/mealybugs",
        "prevention": "Monitor and control sap-sucking insects early",
        "fertilizer": "Maintain tree vigour",
        "severity": "Low",
        "weather": "Dry weather favours insect vectors",
        "insects": "Scale insects, mealybugs, aphids responsible",
    },
    "Papaya_healthy": {
        "cause": "No disease detected",
        "treatment": "Regular care; monitor for ringspot in area",
        "prevention": "Use virus-free planting material",
        "fertilizer": "NPK suitable for papaya",
        "severity": "Low",
        "weather": "Normal",
        "insects": "Monitor aphids and whiteflies",
        "symptoms": "Normal green papaya leaves",
    },
    "Papaya_Anthracnose": {
        "cause": "Fungus (Colletotrichum gloeosporioides)",
        "symptoms": "Sunken dark spots on fruit; leaf spotting",
        "treatment": "Apply copper or mancozeb fungicide; remove infected fruit",
        "prevention": "Good orchard sanitation; avoid fruit injury",
        "fertilizer": "Balanced papaya nutrition",
        "severity": "Medium",
        "weather": "Wet humid weather",
        "insects": "Not the primary cause",
    },
    "Papaya_Bacterial_spot": {
        "cause": "Bacterial leaf spot on papaya",
        "symptoms": "Water-soaked angular spots on leaves",
        "treatment": "Apply copper bactericide; improve air circulation",
        "prevention": "Avoid overhead watering; remove infected leaves",
        "fertilizer": "Balanced nutrition",
        "severity": "Medium",
        "weather": "Warm wet weather",
        "insects": "Spread by rain splash",
    },
    "Papaya_Leaf_curl": {
        "cause": "Virus (Papaya leaf curl virus) spread by whitefly",
        "symptoms": "Leaf curling, distortion, vein thickening",
        "treatment": "No cure — remove infected plants; control whitefly vectors",
        "prevention": "Whitefly management; reflective mulch; resistant varieties",
        "fertilizer": "Support healthy plants",
        "severity": "Medium",
        "weather": "Dry warm weather favours whitefly",
        "insects": "Whitefly spreads the virus",
    },
    "Papaya_Ringspot": {
        "cause": "Virus (Papaya ringspot virus — PRSV)",
        "symptoms": "Ring patterns and distortion on young leaves",
        "treatment": "No cure — remove infected plants; control aphid vectors",
        "prevention": "Resistant varieties where available; vector control",
        "fertilizer": "Support healthy plants only",
        "severity": "High",
        "weather": "Spreads with aphid activity",
        "insects": "Aphids spread the virus",
    },
    "Tea_healthy": {
        "cause": "No disease detected",
        "treatment": "Standard tea field nutrition and plucking",
        "prevention": "Good drainage; shade management",
        "fertilizer": "Tea fertilizer schedule per TRI guidelines",
        "severity": "Low",
        "weather": "Normal",
        "insects": "Monitor tea pests",
        "symptoms": "Normal green tea leaves",
    },
    "Tea_Algal_leaf_spot": {
        "cause": "Alga (Cephaleuros virescens)",
        "symptoms": "Orange-brown velvety patches on leaf surface",
        "treatment": "Improve drainage and air circulation; copper fungicide if severe",
        "prevention": "Prune for airflow; avoid dense shade",
        "fertilizer": "Follow TRI fertilizer recommendations",
        "severity": "Low",
        "weather": "Wet humid weather",
        "insects": "Not caused by insects",
    },
    "Tea_Anthracnose": {
        "cause": "Fungus (Colletotrichum camelliae)",
        "symptoms": "Irregular brown necrotic spots on leaves; dieback of young shoots",
        "treatment": "Apply copper or mancozeb fungicide; prune affected branches",
        "prevention": "Avoid overhead irrigation; maintain balanced nutrition",
        "fertilizer": "Per TRI guidelines",
        "severity": "Medium",
        "weather": "Wet humid conditions",
        "insects": "Not the primary cause",
    },
    "Tea_Bird_eye_spot": {
        "cause": "Fungus (Cercospora theae)",
        "symptoms": "Small circular spots with gray centre and dark border — resembling bird's eye",
        "treatment": "Apply copper fungicide; improve drainage; reduce shade",
        "prevention": "Field sanitation; balanced fertilizer",
        "fertilizer": "Per TRI guidelines",
        "severity": "Medium",
        "weather": "Wet weather",
        "insects": "Not caused by insects",
    },
    "Tea_Brown_blight": {
        "cause": "Fungal blight on tea leaves (Colletotrichum / Glomerella species)",
        "symptoms": "Brown necrotic patches on mature leaves",
        "treatment": "Fungicide per Tea Research Institute advice; improve drainage",
        "prevention": "Pruning; balanced fertilizer",
        "fertilizer": "As per local tea extension",
        "severity": "Medium",
        "weather": "Prolonged wet weather",
        "insects": "Not the primary cause",
    },
    "Tea_Gray_blight": {
        "cause": "Fungus (Pestalozzia theae / Pestalotiopsis species)",
        "symptoms": "Grayish irregular spots with dark rings; leaf drop",
        "treatment": "Apply copper or carbendazim fungicide; prune affected foliage",
        "prevention": "Avoid excess nitrogen; improve field drainage",
        "fertilizer": "Per TRI guidelines",
        "severity": "Medium",
        "weather": "Cool wet periods",
        "insects": "Not caused by insects",
    },
    "Tea_Red_rust": {
        "cause": "Alga (Cephaleuros parasiticus)",
        "symptoms": "Orange-red rust-like patches on both leaf surfaces",
        "treatment": "Copper fungicide sprays; improve aeration and drainage",
        "prevention": "Prune for canopy openness; avoid waterlogging",
        "fertilizer": "Balanced tea fertilizer schedule",
        "severity": "Low",
        "weather": "Wet humid conditions",
        "insects": "Not caused by insects",
    },
    "Tea_White_spot": {
        "cause": "Fungus (Phyllosticta theae)",
        "symptoms": "White or grey circular spots with brown border on leaves",
        "treatment": "Apply copper fungicide; improve field hygiene",
        "prevention": "Remove infected leaves; reduce shade",
        "fertilizer": "Per TRI guidelines",
        "severity": "Low",
        "weather": "Wet weather",
        "insects": "Not caused by insects",
    },
    "Chili_healthy": {
        "cause": "No disease detected",
        "treatment": "Continue regular chili field management",
        "prevention": "Crop rotation; clean seed",
        "fertilizer": "Balanced NPK for chili",
        "severity": "Low",
        "weather": "Normal",
        "insects": "Monitor thrips and mites",
        "symptoms": "Normal green chili leaves",
    },
    "Chili_Bacterial_spot": {
        "cause": "Bacterial leaf spot on chili-type peppers (Xanthomonas)",
        "symptoms": "Water-soaked spots on leaves and fruit",
        "treatment": "Copper sprays; use certified seed; avoid working wet fields",
        "prevention": "Resistant varieties; crop rotation",
        "fertilizer": "Balanced nutrition",
        "severity": "Medium",
        "weather": "Warm wet weather",
        "insects": "Can spread on tools and splash",
    },
    "Coconut_healthy": {
        "cause": "No disease detected",
        "treatment": "Regular coconut estate management",
        "prevention": "Sanitation; pest monitoring",
        "fertilizer": "NPK for coconut as per CRI advice",
        "severity": "Low",
        "weather": "Normal",
        "insects": "Monitor rhinoceros beetle and mites",
        "symptoms": "Normal green coconut fronds",
    },
    "Coconut_Leaf_rot": {
        "cause": "Leaf rot / spotting on coconut palm foliage — fungal complex",
        "symptoms": "Brown rotting areas on fronds",
        "treatment": "Remove severely affected fronds; improve drainage; consult CRI Lunuwila",
        "prevention": "Good nutrition; avoid waterlogging",
        "fertilizer": "Potassium and micronutrients per soil test",
        "severity": "Medium",
        "weather": "Very wet conditions",
        "insects": "Secondary pests may follow damage",
    },
    "Coconut_Gray_leaf_spot": {
        "cause": "Gray leaf spot on coconut palm — fungal infection",
        "symptoms": "Yellow-gray spots on frond leaflets",
        "treatment": "Remove affected fronds; improve nutrition; consult CRI Lunuwila",
        "prevention": "Balanced fertilizer; avoid water stress",
        "fertilizer": "Potassium and micronutrients per soil test",
        "severity": "Medium",
        "weather": "Humid periods",
        "insects": "Monitor for secondary pests",
    },
}

# ---------------------------------------------------------------------------
# Helper: find the best DISEASE_INFO key from a (plant_type, disease) pair
# ---------------------------------------------------------------------------
def _find_disease_key(plant_type, disease_name):
    """Find the best matching DISEASE_INFO key."""
    if not plant_type:
        return disease_name if disease_name in DISEASE_INFO else None

    plant_t = plant_type.strip().capitalize()
    disease_n = disease_name.strip()

    if disease_n.lower() == "healthy":
        key = f"{plant_t}_healthy"
        if key in DISEASE_INFO:
            return key

    key = f"{plant_t}_{disease_n}"
    if key in DISEASE_INFO:
        return key

    key = f"{plant_t}_{disease_n.replace(' ', '_')}"
    if key in DISEASE_INFO:
        return key

    lower_key = key.lower()
    for k in DISEASE_INFO:
        if k.lower() == lower_key:
            return k

    # Handle PlantVillage triple-underscore (Apple___Apple_scab)
    pv_key = f"{plant_t}___{disease_n}"
    if pv_key in DISEASE_INFO:
        return pv_key
    pv_key2 = f"{plant_t}___{disease_n.replace(' ', '_')}"
    if pv_key2 in DISEASE_INFO:
        return pv_key2

    for k in DISEASE_INFO:
        if k.lower().startswith(plant_t.lower()):
            k_norm = k.replace("_", " ").replace("(", "").replace(")", "").lower()
            d_norm = (
                disease_n.replace("_", " ")
                .replace("(", "")
                .replace(")", "")
                .lower()
            )
            words = d_norm.split()
            if len(words) >= 2:
                if all(w in k_norm for w in words):
                    return k
            elif d_norm in k_norm:
                return k

    return None


# ---------------------------------------------------------------------------
# PyTorch MobileNetV3-Large — lazy-loaded
# ---------------------------------------------------------------------------
_MOBILENET_MODEL = None
_MOBILENET_CLASSES = None
_MOBILENET_DEVICE = None


def _load_mobilenet_model():
    global _MOBILENET_MODEL, _MOBILENET_CLASSES, _MOBILENET_DEVICE

    if _MOBILENET_MODEL is not None:
        return _MOBILENET_MODEL, _MOBILENET_CLASSES, _MOBILENET_DEVICE

    try:
        import torch
        from torchvision import transforms, models, datasets

        backend_dir = Path(__file__).resolve().parent
        model_path = backend_dir.parent / "Merge-Project" / "output" / "plant_disease_mobilenet.pth"
        data_dir = backend_dir.parent / "Merge-Project" / "resized_merged"

        if not model_path.exists() or not data_dir.is_dir():
            _MOBILENET_MODEL = None
            return None, None, None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = datasets.ImageFolder(str(data_dir))
        # Remove 'test' folder if present (matches training script behaviour)
        if "test" in dataset.classes:
            test_idx = dataset.class_to_idx["test"]
            dataset.samples = [s for s in dataset.samples if s[1] != test_idx]
            dataset.targets = [s[1] for s in dataset.samples]
            dataset.class_to_idx = {k: i for i, (k, v) in enumerate(
                [(c, i) for c, i in dataset.class_to_idx.items() if c != "test"]
            )}
            dataset.classes = [c for c in dataset.classes if c != "test"]
        class_names = dataset.classes
        num_classes = len(class_names)

        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model = model.to(device)
        model.eval()

        _MOBILENET_MODEL = model
        _MOBILENET_CLASSES = class_names
        _MOBILENET_DEVICE = device
        return model, class_names, device

    except Exception as e:
        print(f"MobileNet model load failed: {e}")
        return None, None, None


def _predict_with_pytorch(image_bytes):
    """
    Run inference using the trained MobileNetV3-Large model.
    Returns (class_name, confidence_0_100, plant_type) or None.
    """
    model, class_names, device = _load_mobilenet_model()
    if model is None:
        return None

    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        import io

        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probs, k=2, dim=1)
            conf_pct = round(float(top_probs[0][0].item()) * 100, 1)
            pred_class = class_names[top_indices[0][0].item()]

        from leaf_analysis import _class_crop
        pt = _class_crop(pred_class) or "unknown"

        return pred_class, conf_pct, pt

    except Exception as e:
        print(f"PyTorch predict failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Image bytes → OpenCV
# ---------------------------------------------------------------------------
def _bytes_to_cv2(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# DISEASE_INFO builder
# ---------------------------------------------------------------------------
def _build_info(disease_key):
    if disease_key and disease_key in DISEASE_INFO:
        base = DISEASE_INFO[disease_key]
        return {
            "display_name": disease_key.replace("_", " "),
            "symptoms": base.get("symptoms", ""),
            "treatment": base.get("treatment", "Consult local agricultural expert"),
            "cause": base.get("cause", "Unknown"),
            "weather": base.get("weather", ""),
            "insects": base.get("insects", ""),
            "prevention": base.get("prevention", ""),
            "fertilizer": base.get("fertilizer", ""),
            "severity": base.get("severity", "Medium"),
        }
    return None


_UNIDENTIFIED_INFO = {
    "display_name": "Unidentified",
    "symptoms": "",
    "treatment": "The image does not contain a plant leaf. Please upload a clear photo of a leaf.",
    "cause": "No plant leaf detected",
    "severity": "Unknown",
    "prevention": "",
    "weather": "",
    "insects": "",
    "fertilizer": "",
}


# ---------------------------------------------------------------------------
# Random Forest model — lazy-loaded
# ---------------------------------------------------------------------------
_RF_CACHE = {"loaded": False, "model": None, "scaler": None, "encoder": None}


def _load_rf_model():
    if _RF_CACHE["loaded"]:
        return _RF_CACHE["model"], _RF_CACHE["scaler"], _RF_CACHE["encoder"]
    try:
        import joblib
        from config import MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH

        _RF_CACHE["model"] = joblib.load(MODEL_PATH)
        _RF_CACHE["scaler"] = joblib.load(SCALER_PATH)
        _RF_CACHE["encoder"] = joblib.load(LABEL_ENCODER_PATH)
        _RF_CACHE["loaded"] = True
        _log("RF", "Random Forest model loaded")
    except Exception as e:
        _log("RF", f"Failed to load Random Forest model: {e}")
        return None, None, None
    return _RF_CACHE["model"], _RF_CACHE["scaler"], _RF_CACHE["encoder"]


def extract_features(img):
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    features.append(edges.mean())
    for i in range(3):
        features.append(img[:, :, i].mean())
        features.append(img[:, :, i].std())
    return np.array(features)


def _predict_with_rf(image_bytes, opencv_crop=None):
    model, scaler, encoder = _load_rf_model()
    if model is None:
        return None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        features = extract_features(img)
        features_scaled = scaler.transform([features])
        proba = model.predict_proba(features_scaled)[0]
        best_idx = int(np.argmax(proba))
        disease = str(encoder.classes_[best_idx])
        confidence = round(float(proba[best_idx]) * 100, 1)

        from leaf_analysis import _class_crop

        # If OpenCV detected a crop with high confidence, respect it
        if opencv_crop:
            opencv_crop_lower = opencv_crop.lower()
            # Check if any class matching the detected crop has decent probability
            crop_classes = [
                (i, float(proba[i]))
                for i, c in enumerate(encoder.classes_)
                if str(c).lower().startswith(opencv_crop_lower)
            ]
            if crop_classes:
                best_crop_i, best_crop_p = max(crop_classes, key=lambda x: x[1])
                best_crop_disease = str(encoder.classes_[best_crop_i])
                best_crop_conf = round(best_crop_p * 100, 1)
                # Use crop-filtered result if it's the same as raw best OR has >= 5%
                if best_crop_conf >= 5.0:
                    disease = best_crop_disease
                    confidence = best_crop_conf
                    best_idx = best_crop_i
                elif best_crop_conf < 5.0 and confidence < 8.0:
                    # Neither raw nor crop-filtered is confident enough
                    _log("RF", f"Low confidence for crop={opencv_crop} (best={best_crop_conf}%) and overall (top={confidence}%)")
                    return None

        top_idx = np.argsort(proba)[::-1][:5]
        top_predictions = [
            {
                "disease": str(encoder.classes_[i]),
                "confidence": round(float(proba[i]) * 100, 1),
            }
            for i in top_idx
        ]
        plant_type = _class_crop(disease) or disease.split("_")[0]
        return disease, confidence, plant_type, top_predictions
    except Exception as e:
        _log("RF", f"RF predict failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Histogram quality check
# ---------------------------------------------------------------------------
def _check_histogram_quality(img: np.ndarray) -> tuple[bool, str, dict]:
    """
    Check image histogram for basic quality.
    Returns (pass, message, histogram_metrics).
    """
    h, w = img.shape[:2]
    total_px = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / max(total_px, 1)

    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    # Fraction of pixels in the most populated bin
    peak_bin = float(np.max(hist))
    # Number of bins with non-zero pixels
    nonzero_bins = int(np.count_nonzero(hist))
    # Dark pixel fraction (0-30)
    dark_frac = float(np.sum(gray < 30)) / total_px
    # Bright pixel fraction (220-255)
    bright_frac = float(np.sum(gray > 220)) / total_px

    hist_metrics = {
        "mean_brightness": round(mean_brightness, 1),
        "std_brightness": round(std_brightness, 1),
        "peak_bin_fraction": round(float(peak_bin), 4),
        "nonzero_bins": nonzero_bins,
        "dark_fraction": round(dark_frac, 3),
        "bright_fraction": round(bright_frac, 3),
    }

    messages = []

    # Too dark
    if mean_brightness < 20:
        messages.append("image too dark")
    # Too bright / washed out
    elif mean_brightness > 240:
        messages.append("image too bright (washed out)")
    # Very low contrast (std < 15)
    if std_brightness < 15:
        messages.append("low contrast image")
    # Nearly uniform (peak bin > 80% of pixels)
    if peak_bin > 0.80:
        messages.append("near-uniform image (solid color)")

    passed = len(messages) == 0
    msg = "; ".join(messages) if messages else "histogram OK"
    return passed, msg, hist_metrics


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT — 5-stage pipeline
# ---------------------------------------------------------------------------
def predict_disease(image_bytes, farmer_context=None):
    """
    Pipeline:
      1. OpenCV leaf gate        — reject non-leaf images immediately
      2. Histogram quality check  — verify image has sufficient content
      3. Gemini API               — MAIN predictor (vision model)
      4. MobileNetV3-Large       — fallback if Gemini fails / returns unknown
      5. Random Forest            — backup analysis only (supplementary metadata)
    """
    _log("INPUT", f"Image: {len(image_bytes)} bytes")

    from leaf_analysis import detect_crop_family, is_likely_a_leaf

    img = _bytes_to_cv2(image_bytes)
    if img is None:
        return "unidentified", 0.0, dict(_UNIDENTIFIED_INFO), [], "unknown", {"source": "invalid_image"}

    # ── STAGE 1: OpenCV leaf gate ──
    _log("Stage 1", "OpenCV leaf gate...")
    cv_crop, cv_conf, metrics = detect_crop_family(img)
    is_leaf, reject_reason = is_likely_a_leaf(metrics)
    _log("Stage 1", f"  Leaf? {is_leaf}  reason='{reject_reason}'  crop={cv_crop}")

    if not is_leaf:
        _log("Stage 1", "REJECTED — not a leaf")
        return "unidentified", 0.0, dict(_UNIDENTIFIED_INFO), [], "unknown", {"source": "leaf_gate", "reason": reject_reason}

    _log("Stage 1", "PASSED — leaf detected")

    # ── STAGE 2: Histogram quality check ──
    _log("Stage 2", "Histogram quality check...")
    hist_ok, hist_msg, hist_metrics = _check_histogram_quality(img)
    _log("Stage 2", f"  Histogram: {hist_msg}  metrics={hist_metrics}")
    if not hist_ok:
        _log("Stage 2", f"  Warning: {hist_msg} — proceeding anyway")

    # ── STAGE 3: MobileNet prediction ──
    _log("Stage 3", "MobileNet prediction...")
    mobilenet_result = _predict_with_pytorch(image_bytes)
    if mobilenet_result:
        mn_disease, mn_conf, mn_plant = mobilenet_result
        _log("Stage 3", f"  MobileNet: '{mn_disease}' @ {mn_conf}%  plant='{mn_plant}'")
        if mn_conf >= 15:
            disease_key = _find_disease_key(mn_plant.capitalize(), mn_disease)
            info = _build_info(disease_key)
            top = [{"disease": mn_disease, "confidence": mn_conf}]

            rf_supplementary = None
            try:
                rf_res = _predict_with_rf(image_bytes, opencv_crop=cv_crop)
                if rf_res:
                    _, rf_conf, rf_plant, rf_top = rf_res
                    rf_supplementary = {
                        "top_predictions": rf_top,
                        "plant_type": rf_plant,
                        "confidence": rf_conf,
                    }
            except Exception:
                pass

            meta = {
                "source": "mobilenet",
                "plant_type": mn_plant,
                "opencv_crop": cv_crop,
                "histogram": hist_metrics,
                "rf_backup": rf_supplementary,
            }

            _log("Result", f"'{mn_disease}' @ {mn_conf}% (source: mobilenet)")
            return mn_disease, mn_conf, info or dict(_UNIDENTIFIED_INFO), top, mn_plant, meta

    # MobileNet failed
    _log("Result", "MobileNet failed — returning unidentified")
    return "unidentified", 0.0, dict(_UNIDENTIFIED_INFO), [], "unknown", {"source": "mobilenet_failed", "opencv_crop": cv_crop}
