"""
Computer-vision heuristics to detect crop type and lesion patterns from leaf photos.
Used to re-rank the sklearn model so crops are not confused (e.g. grape vs corn).
"""

from __future__ import annotations

import cv2
import numpy as np

# PlantVillage-style class prefixes
CROP_PREFIXES = (
    "Banana",
    "Rice",
    "Coconut",
    "Tea",
    "Chili",
    "Mango",
    "Papaya",
    "Apple",
    "Blueberry",
    "Cherry_(including_sour)",
    "Corn_(maize)",
    "Grape",
    "Orange",
    "Peach",
    "Pepper,_bell",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
)


def _leaf_mask(img: np.ndarray) -> np.ndarray: 
    """Binary mask of green leaf pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 35, 35]), np.array([95, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _lobe_count(contour) -> int:
    """Deep concavities on the leaf outline — grape leaves are deeply lobed."""
    if contour is None or len(contour) < 5:
        return 0
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return 0
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return 0
    if defects is None:
        return 0
    perim = max(cv2.arcLength(contour, True), 1.0)
    count = 0
    for i in range(defects.shape[0]):
        _, _, _, depth = defects[i, 0]
        if depth / 256.0 > 0.035 * perim:
            count += 1
    return count


def _edge_serration_score(mask: np.ndarray, contour) -> float:
    """
    Measure how serrated (toothed/jagged) the leaf edge is.
    Returns a score 0.0 (smooth) to ~1.0 (very serrated).
    Apple leaves have serrated edges (high score); mango leaves have smooth edges (low score).
    """
    if contour is None or cv2.contourArea(contour) < 200:
        return 0.0
    # Sample points along contour
    perim = cv2.arcLength(contour, True)
    if perim < 20:
        return 0.0
    # Get contour points densely
    epsilon = 0.001 * perim
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 10:
        return 0.0
    # Measure jaggedness: compute distances from each point to a smoothed local line
    pts = approx[:, 0, :]
    n = len(pts)
    # Use curvature variance: compute angle changes along contour
    angles = []
    for i in range(n):
        p0 = pts[(i - 2) % n]
        p1 = pts[i]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1 or norm2 < 1:
            continue
        cos_a = np.dot(v1, v2) / (norm1 * norm2)
        cos_a = max(-1.0, min(1.0, cos_a))
        angles.append(np.arccos(cos_a))
    if len(angles) < 5:
        return 0.0
    # High variance in angle = jagged/serrated edge
    angle_std = float(np.std(angles))
    # Normalize: typical smooth leaf ~0.3-0.5, serrated ~0.8-2.0
    serration = min(1.0, angle_std / 1.5)
    return serration


def _lesion_stats(img: np.ndarray, mask: np.ndarray) -> dict:
    """Brown/orange lesion area and roundness on the leaf."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, np.array([5, 45, 45]), np.array([28, 255, 220]))
    yellow = cv2.inRange(hsv, np.array([15, 40, 70]), np.array([42, 255, 255]))
    brown_m = cv2.bitwise_and(brown, brown, mask=mask)
    yellow_m = cv2.bitwise_and(yellow, yellow, mask=mask)
    damage = cv2.bitwise_or(brown_m, yellow_m)
    leaf_px = max(int(np.sum(mask > 0)), 1)
    lesion_px = int(np.sum(brown_m > 0))
    ratio = lesion_px / leaf_px
    yellow_ratio = int(np.sum(yellow_m > 0)) / leaf_px
    damage_ratio = int(np.sum(damage > 0)) / leaf_px
    lesion = damage

    round_spots = 0
    if lesion_px > 80:
        contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 40:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity > 0.45:
                round_spots += 1

    elongated_patches = 0
    if lesion_px > 80:
        contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 120:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            patch_aspect = max(bw, bh) / max(min(bw, bh), 1)
            if patch_aspect >= 2.2:
                elongated_patches += 1

    has_circular = round_spots >= 2 and elongated_patches <= 1
    if round_spots >= 1 and ratio > 0.08 and elongated_patches == 0:
        has_circular = True

    return {
        "lesion_ratio": ratio,
        "yellow_ratio": round(float(yellow_ratio), 4),
        "damage_ratio": round(float(damage_ratio), 4),
        "round_spots": round_spots,
        "elongated_patches": elongated_patches,
        "has_circular_lesions": has_circular,
        "has_elongated_lesions": elongated_patches >= 1,
    }


def has_heavy_leaf_damage(metrics: dict) -> bool:
    """Obvious disease on leaf — must not report 'healthy'."""
    damage = float(metrics.get("damage_ratio", metrics.get("lesion_ratio", 0.0)))
    yellow = float(metrics.get("yellow_ratio", 0.0))
    return (
        damage >= 0.06
        or yellow >= 0.05
        or bool(metrics.get("has_elongated_lesions"))
        or int(metrics.get("elongated_patches", 0)) >= 2
        or int(metrics.get("round_spots", 0)) >= 4
    )


def has_visible_leaf_disease(metrics: dict) -> bool:
    """Visible spots/streaks on phone photos (lower bar than heavy damage)."""
    if has_heavy_leaf_damage(metrics):
        return True
    lesion = float(metrics.get("lesion_ratio", 0.0))
    yellow = float(metrics.get("yellow_ratio", 0.0))
    return lesion >= 0.035 or yellow >= 0.028 or int(metrics.get("elongated_patches", 0)) >= 1


def _metrics_coconut_frond(metrics: dict) -> bool:
    """Coconut/palm frond shape from metrics only (avoids recursion)."""
    lobes = int(metrics.get("lobes", 0))
    if lobes >= 3:
        return False
    aspect = float(metrics.get("aspect", 1.0))
    if aspect >= 2.25:
        return False
    lesion_ratio = float(metrics.get("lesion_ratio", 0.0))
    round_spots = int(metrics.get("round_spots", 0))
    long_blade = (1.4 <= aspect <= 2.2) or aspect <= 0.85
    heavy_rust = lesion_ratio >= 0.06 and (
        round_spots >= 1 or lesion_ratio >= 0.10
    )
    return long_blade and heavy_rust


def _metrics_mango_oval(metrics: dict) -> bool:
    """Mango oval / lanceolate shape from metrics only (avoids recursion)."""
    lobes = int(metrics.get("lobes", 0))
    if lobes >= 3:
        return False
    aspect = float(metrics.get("aspect", 1.0))
    damage = float(metrics.get("damage_ratio", metrics.get("lesion_ratio", 0.0)))
    solidity = float(metrics.get("solidity", 1.0))
    round_spots = int(metrics.get("round_spots", 0))
    elongated = bool(metrics.get("has_elongated_lesions"))
    yellow = float(metrics.get("yellow_ratio", 0.0))
    if elongated and yellow >= 0.20:
        return False
    if (
        lobes <= 1
        and 0.95 <= aspect <= 2.65
        and 0.004 <= damage <= 0.75
        and solidity >= 0.85
        and not elongated
        and yellow < 0.20
    ):
        return True
    if round_spots >= 2 and 0.9 <= aspect <= 2.5 and lobes <= 1:
        return True
    if (
        round_spots >= 3
        and 0.35 <= aspect <= 4.5
        and lobes <= 2
        and yellow < 0.15
        and not elongated
    ):
        return True
    if round_spots >= 4 and 0.004 <= damage <= 0.75 and lobes <= 2 and yellow < 0.18:
        return True
    return False


def _metrics_mango_spotted_blade(metrics: dict) -> bool:
    """Mango with round gall/anthracnose spots — not banana Sigatoka streaks."""
    if is_banana_sigatoka_streak_pattern(metrics):
        return False
    round_spots = int(metrics.get("round_spots", 0))
    yellow = float(metrics.get("yellow_ratio", 0.0))
    aspect = float(metrics.get("aspect", 1.0))
    lobes = int(metrics.get("lobes", 0))
    if round_spots >= 3 and yellow < 0.15 and lobes <= 2 and 0.35 <= aspect <= 4.5:
        return True
    return _metrics_mango_oval(metrics) or is_mango_leaf(metrics)


def is_banana_sigatoka_streak_pattern(metrics: dict) -> bool:
    """
    Yellow-bordered brown streaks along a strap leaf (Sigatoka).
    Requires elongated lesions — not round rice/mango spot patterns.
    """
    if int(metrics.get("lobes", 0)) >= 3:
        return False
    lesion = float(metrics.get("lesion_ratio", 0.0))
    yellow = float(metrics.get("yellow_ratio", 0.0))
    elongated = bool(metrics.get("has_elongated_lesions")) or int(
        metrics.get("elongated_patches", 0)
    ) >= 1
    round_spots = int(metrics.get("round_spots", 0))
    aspect = float(metrics.get("aspect", 1.0))
    if _metrics_coconut_frond(metrics):
        return False
    if round_spots >= 3 and not elongated:
        return False
    if not elongated:
        return False
    damage = float(metrics.get("damage_ratio", metrics.get("lesion_ratio", 0.0)))
    strap = aspect >= 1.35 or aspect <= 0.72
    if lesion >= 0.08 and yellow >= 0.04 and strap:
        return True
    if lesion >= 0.14 and yellow >= 0.06:
        return True
    if (
        yellow >= 0.25
        and elongated
        and round_spots <= 2
        and lesion >= 0.02
        and damage >= 0.15
    ):
        return True
    if (
        yellow >= 0.12
        and elongated
        and round_spots <= 2
        and lesion >= 0.04
        and damage >= 0.08
    ):
        return True
    if int(metrics.get("elongated_patches", 0)) >= 5 and round_spots == 0:
        return False
    return False


def _metrics_banana_strap(metrics: dict) -> bool:
    """Long banana/plantain blade from metrics only (no is_* cross-calls)."""
    if _metrics_coconut_frond(metrics) or _metrics_mango_oval(metrics):
        return False
    if _metrics_mango_spotted_blade(metrics):
        return False
    round_spots = int(metrics.get("round_spots", 0))
    if round_spots >= 3 and float(metrics.get("yellow_ratio", 0.0)) < 0.15:
        if not is_banana_sigatoka_streak_pattern(metrics):
            return False
    lobes = int(metrics.get("lobes", 0))
    aspect = float(metrics.get("aspect", 1.0))
    if lobes > 2:
        return False
    yellow = float(metrics.get("yellow_ratio", 0.0))
    elongated = bool(metrics.get("has_elongated_lesions"))
    damage = float(metrics.get("damage_ratio", metrics.get("lesion_ratio", 0.0)))
    if (
        0.9 <= aspect <= 2.7
        and float(metrics.get("solidity", 1.0)) >= 0.88
        and not elongated
        and yellow < 0.04
        and damage < 0.05
    ):
        return False
    if elongated and damage >= 0.06 and yellow >= 0.05:
        if aspect >= 2.4 or aspect <= 0.40:
            return True
    if aspect >= 2.85 and elongated and damage >= 0.05:
        return True
    if (aspect <= 0.5 or aspect >= 3.0) and yellow >= 0.06 and elongated:
        return True
    if aspect >= 3.0 and has_heavy_leaf_damage(metrics):
        return True
    return False


def looks_like_banana_sigatoka(metrics: dict) -> bool:
    """
    Banana / plantain strap with Sigatoka-style streak lesions only.
    """
    if _metrics_mango_oval(metrics) and not is_banana_sigatoka_streak_pattern(metrics):
        return False
    if is_banana_sigatoka_streak_pattern(metrics):
        return True
    return _metrics_banana_strap(metrics)


def is_confirmed_banana_leaf(metrics: dict) -> bool:
    """True only when shape + lesions clearly indicate banana (not generic damage)."""
    if _metrics_mango_spotted_blade(metrics):
        return False
    if is_mango_leaf(metrics) and not is_banana_sigatoka_streak_pattern(metrics):
        return False
    if is_banana_sigatoka_streak_pattern(metrics):
        return True
    return _metrics_banana_strap(metrics)


def detect_crop_family(img: np.ndarray) -> tuple[str | None, float, dict]:
    """
    Guess crop family from leaf shape.
    Returns (crop_prefix, confidence 0-1, debug metrics).
    """
    work = cv2.resize(img, (256, 256))
    mask = _leaf_mask(work)
    contour = _largest_contour(mask)
    green_px = int(np.sum(mask > 0))
    total_px = work.shape[0] * work.shape[1]  # 256*256
    # Count distinct green regions (real leaf = 1 main region; background = many)
    all_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = sum(1 for c in all_contours if cv2.contourArea(c) > 100)
    metrics: dict = {"lobes": 0, "aspect": 1.0, "solidity": 1.0, "serration": 0.0, "compactness": 0.0, "area": 0.0, "has_green": green_px > 2000, "green_ratio": green_px / max(total_px, 1), "contour_points": 0, "extent": 1.0, "num_regions": num_regions, "border_touching": False}

    scores = {p: 0.0 for p in CROP_PREFIXES}

    if contour is not None and cv2.contourArea(contour) > 500:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        metrics["aspect"] = round(aspect, 2)

        hull = cv2.convexHull(contour)
        hull_area = max(cv2.contourArea(hull), 1.0)
        area_val = cv2.contourArea(contour)
        metrics["area"] = round(float(area_val), 1)
        solidity = area_val / hull_area
        metrics["solidity"] = round(float(solidity), 2)

        lobes = _lobe_count(contour)
        metrics["lobes"] = lobes

        # ── Edge serration ── key for apple vs mango
        serration = _edge_serration_score(mask, contour)
        metrics["serration"] = round(serration, 3)

        # Compactness: (perimeter^2) / (4 * pi * area); lower = rounder
        perim = cv2.arcLength(contour, True)
        compactness = (perim * perim) / (4 * np.pi * max(area_val, 1.0))
        metrics["compactness"] = round(float(compactness), 2)

        # Extent: contour area / bounding rect area; high = fills box (rectangular)
        rect_area = w * h
        extent = area_val / max(rect_area, 1)
        metrics["extent"] = round(float(extent), 2)

        # Number of points on the contour; very few = simple/manufactured shape
        metrics["contour_points"] = contour.shape[0]

        # Green ratio: fraction of total image pixels that are green
        metrics["green_ratio"] = round(green_px / max(total_px, 1), 4)

        # Border touching: if the contour touches image edge, likely background not isolated leaf
        h_img, w_img = work.shape[:2]
        touches_border = False
        for pt in contour:
            px, py = pt[0]
            if px <= 2 or px >= w_img - 3 or py <= 2 or py >= h_img - 3:
                touches_border = True
                break
        metrics["border_touching"] = touches_border

        # ── Grape: deeply lobed ──
        if lobes >= 3 and aspect < 2.0:
            scores["Grape"] += 3.0 + min(lobes, 5) * 0.35
        elif lobes >= 2 and aspect < 1.6 and solidity < 0.82:
            scores["Grape"] += 1.0

        # ── Corn: very long narrow grass ──
        if aspect > 2.2 and lobes <= 1:
            scores["Corn_(maize)"] += 1.2
        if aspect > 3.0:
            scores["Corn_(maize)"] += 0.8
        if solidity > 0.92 and aspect > 1.8:
            scores["Corn_(maize)"] += 0.5

        # ── Tomato / Potato ──
        if 1.2 < aspect < 2.0 and lobes <= 1:
            scores["Tomato"] += 1.2
            scores["Potato"] += 1.0

        # ── Apple: oval/round, high solidity, serrated edges ──
        apple_shape = (0.75 <= aspect <= 1.65 and solidity > 0.78 and lobes <= 1)
        if apple_shape:
            scores["Apple"] += 1.5
            if serration > 0.25:  # serrated edges strongly confirms apple
                scores["Apple"] += 2.0
                scores["Mango"] = max(0.0, scores.get("Mango", 0.0) - 1.5)  # penalize mango
            if serration > 0.40:
                scores["Apple"] += 1.5  # very serrated = definitely apple

        # ── Mango: lanceolate (long/narrow), smooth edges, lower solidity ──
        mango_shape = (aspect >= 1.8 and lobes <= 1 and solidity > 0.78)
        if mango_shape:
            scores["Mango"] += 1.5
            if serration < 0.15:  # smooth edges confirm mango
                scores["Mango"] += 2.0
            if aspect >= 2.5:
                scores["Mango"] += 1.5  # very long = very mango
        # Also score mango for medium aspect with palm-like tendency
        if lobes <= 1 and 1.5 <= aspect < 1.8 and solidity > 0.82 and serration < 0.20:
            scores["Mango"] += 1.5

        # ── Banana: large single strap-like blade, smooth edges ──
        banana_portrait = aspect < 0.6 and lobes <= 1 and solidity > 0.85 and serration < 0.15
        banana_landscape = aspect > 1.8 and lobes <= 1 and solidity > 0.85 and serration < 0.15
        if banana_portrait or banana_landscape:
            scores["Banana"] += 3.0
            if aspect > 2.5:
                scores["Banana"] += 1.0  # very wide = very banana
            if aspect < 0.4:
                scores["Banana"] += 1.0  # very tall = very banana
            # Penalize Apple when leaf clearly looks like Banana
            scores["Apple"] = max(0.0, scores.get("Apple", 0.0) - 3.0)

        # ── Pepper ──
        if 1.6 < aspect < 2.5 and lobes <= 1:
            scores["Pepper,_bell"] += 1.2

        # ── Palm-family (Coconut / Papaya) ──
        palm_leaf = lobes <= 2 and (aspect >= 1.55 or aspect <= 0.65)
        if palm_leaf:
            scores["Coconut"] += 2.8
            scores["Papaya"] += 2.2
            if serration < 0.15:
                scores["Coconut"] += 1.5  # smooth edges = palm frond
        if lobes <= 1 and (aspect >= 2.0 or aspect <= 0.5):
            scores["Coconut"] += 0.8
            scores["Papaya"] += 1.5

    lesions = _lesion_stats(work, mask)
    metrics.update(lesions)

    lobes = metrics.get("lobes", 0)

    # ── Suppress lesion-based coconut/papaya boost on apple-shaped leaves ──
    # When serration is high and aspect is round/oval, lesions are apple diseases (scab, rust, rot), NOT coconut/papaya
    is_apple_confirmed = (
        metrics.get("serration", 0.0) > 0.20
        and 0.75 <= metrics.get("aspect", 1.0) <= 1.65
        and metrics.get("solidity", 1.0) > 0.78
        and lobes <= 1
    )

    if lesions.get("has_circular_lesions") and lobes >= 3:
        scores["Grape"] += 1.5
        scores["Tomato"] += 0.3
    elif lesions.get("has_circular_lesions"):
        if not is_apple_confirmed:
            scores["Coconut"] += 1.2
            scores["Papaya"] += 1.0
        scores["Tomato"] += 0.4
    if lesions.get("has_elongated_lesions"):
        if not is_apple_confirmed:
            scores["Coconut"] += 1.2
            scores["Papaya"] += 0.8
        scores["Banana"] += 0.6

    if (
        not is_apple_confirmed
        and metrics.get("lobes", 0) < 3
        and metrics.get("lesion_ratio", 0) >= 0.05
    ):
        scores["Coconut"] += 2.0 + min(metrics.get("lesion_ratio", 0) * 8, 2.5)
        scores["Grape"] *= 0.0
    elif metrics.get("lobes", 0) < 3:
        scores["Grape"] = 0.0

    best_crop = max(scores, key=scores.get)
    best_score = scores[best_crop]
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0

    if best_score < 1.5:
        return None, 0.0, metrics

    confidence = min(0.95, 0.45 + (best_score - second) * 0.12 + best_score * 0.05)
    return best_crop, confidence, metrics


def _class_crop(class_name: str) -> str | None:
    for prefix in CROP_PREFIXES:
        if class_name.startswith(prefix):
            return prefix
    return None


def refine_probabilities(
    img: np.ndarray,
    classes: np.ndarray,
    proba: np.ndarray,
) -> tuple[int, np.ndarray, str | None, dict]:
    """
    Re-weight class probabilities using detected crop and lesion pattern.
    Returns (best_index, adjusted_proba, crop_family, metrics).
    """
    crop, crop_conf, metrics = detect_crop_family(img)
    adjusted = proba.copy().astype(float)

    max_ml_conf = float(np.max(proba)) if len(proba) > 0 else 0.0
    if max_ml_conf < 0.08:
        best_idx = int(np.argmax(adjusted))
        return best_idx, adjusted, None, metrics

    lobes = int(metrics.get("lobes", 0))
    banana_ml = crop_max_proba(classes, proba, "Banana")
    if (
        not _metrics_mango_spotted_blade(metrics)
        and (is_banana_sigatoka_streak_pattern(metrics) or is_banana_leaf(metrics))
    ):
        crop = "Banana"
        crop_conf = max(crop_conf, 0.94)
    elif (
        banana_ml >= 0.12
        and banana_ml >= crop_max_proba(classes, proba, "Mango")
        and not _metrics_mango_spotted_blade(metrics)
        and (is_banana_sigatoka_streak_pattern(metrics) or _metrics_banana_strap(metrics))
    ):
        crop = "Banana"
        crop_conf = max(crop_conf, 0.82)
    elif _raw_coconut_palm_shape(metrics):
        crop = "Coconut"
        crop_conf = max(crop_conf, 0.92)
    elif is_mango_leaf(metrics):
        crop = "Mango"
        crop_conf = max(crop_conf, 0.88)
    elif _metrics_coconut_frond(metrics):
        crop = "Coconut"
        crop_conf = max(crop_conf, 0.9)

    if crop == "Grape" and lobes < 3:
        crop = "Coconut" if metrics.get("aspect", 1.0) >= 1.4 or metrics.get("aspect", 1.0) <= 0.7 else crop
        if crop == "Grape":
            crop_conf *= 0.35

    # Conservative re-weighting: require decent crop_conf and apply milder multipliers
    if crop and crop_conf >= 0.6:
        for i, name in enumerate(classes):
            cls_crop = _class_crop(str(name))
            if cls_crop == crop:
                adjusted[i] *= 1.0 + crop_conf * 1.2
            elif cls_crop and cls_crop != crop:
                adjusted[i] *= max(0.10, 1.0 - crop_conf * 0.6)

    if lobes < 3:
        for i, name in enumerate(classes):
            if str(name).startswith("Grape_"):
                adjusted[i] *= 0.25

    palm_signal = lobes <= 2 and (
        metrics.get("aspect", 1.0) >= 1.4
        or metrics.get("aspect", 1.0) <= 0.7
        or metrics.get("has_elongated_lesions")
    )
    if palm_signal and is_banana_sigatoka_streak_pattern(metrics):
        for i, name in enumerate(classes):
            if str(name).startswith("Banana_"):
                adjusted[i] *= 1.4
        for i, name in enumerate(classes):
            if str(name).startswith("Coconut_"):
                adjusted[i] *= 0.5
    elif palm_signal and not is_banana_sigatoka_streak_pattern(metrics):
        for i, name in enumerate(classes):
            n = str(name)
            if n.startswith("Coconut_") or n.startswith("Papaya_"):
                adjusted[i] *= 1.4
    elif is_banana_sigatoka_streak_pattern(metrics):
        for i, name in enumerate(classes):
            if str(name).startswith("Banana_"):
                adjusted[i] *= 1.4
        for i, name in enumerate(classes):
            if str(name).startswith("Coconut_"):
                adjusted[i] *= 0.4
    elif _raw_coconut_palm_shape(metrics):
        for i, name in enumerate(classes):
            n = str(name)
            if n.startswith("Coconut_") or n.startswith("Papaya_"):
                adjusted[i] *= 1.4
        for i, name in enumerate(classes):
            if str(name).startswith("Banana_"):
                adjusted[i] *= 0.5

    if (
        crop == "Grape"
        and lobes >= 3
        and metrics.get("has_circular_lesions")
        and not metrics.get("has_elongated_lesions")
    ):
        for i, name in enumerate(classes):
            n = str(name)
            if n == "Grape_Black_rot":
                adjusted[i] *= 2.0

    try:
        top_ml_idx = int(np.argmax(proba)) if len(proba) > 0 else None
        top_ml_name = str(classes[top_ml_idx]) if top_ml_idx is not None else ""
        top_ml_conf = float(proba[top_ml_idx]) if top_ml_idx is not None else 0.0
        apple_shape = (
            lobes <= 1
            and 0.85 <= float(metrics.get("aspect", 1.0)) <= 1.45
            and float(metrics.get("solidity", 1.0)) >= 0.88
        )
        if apple_shape and top_ml_name.startswith("Blueberry_") and top_ml_conf < 0.28:
            for i, name in enumerate(classes):
                if str(name).startswith("Apple_"):
                    adjusted[i] *= 2.2
            for i, name in enumerate(classes):
                if str(name).startswith("Blueberry_"):
                    adjusted[i] *= 0.45

        # ── FIX 1 ── Apple vs Mango: increase conservatism
        # Apple leaves will only be overridden as Mango when ML is very confident.
        try:
            mango_like_top = top_ml_name.startswith("Mango_")
            if apple_shape and mango_like_top and top_ml_conf < 0.85 and not is_mango_leaf(metrics):
                for i, name in enumerate(classes):
                    n = str(name)
                    if n.startswith("Apple_"):
                        adjusted[i] *= 2.5
                    if n.startswith("Mango_"):
                        adjusted[i] *= 0.35
        except Exception:
            pass
    except Exception:
        pass

    total = adjusted.sum()
    if total > 0:
        adjusted /= total

    best_idx = int(np.argmax(adjusted))
    return best_idx, adjusted, crop, metrics


def detect_unsupported_plant(
    img: np.ndarray,
    metrics: dict,
    *,
    skip_labels: tuple[str, ...] = (),
) -> str | None:
    """
    Detect plants not in the training set (e.g. banana before retrain).
    skip_labels: plants the model already supports (e.g. ("Banana",) after SL training).
    Returns a short plant label or None.
    """
    work = cv2.resize(img, (256, 256))
    mask = _leaf_mask(work)
    contour = _largest_contour(mask)
    if contour is None:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / max(h, 1)
    lobes = metrics.get("lobes", 0)
    solidity = metrics.get("solidity", 1.0)
    lesion_ratio = metrics.get("lesion_ratio", 0.0)
    elongated = metrics.get("has_elongated_lesions", False)

    if "Banana" not in skip_labels:
        if lobes < 3 and elongated and lesion_ratio > 0.04:
            return "Banana"
        banana_shape = (
            lobes <= 1
            and solidity >= 0.80
            and (aspect >= 1.25 or aspect <= 0.75)
        )
        if banana_shape and lesion_ratio > 0.05:
            return "Banana"
        if (aspect > 2.8 or aspect < 0.38) and lobes <= 1 and elongated:
            return "Banana"

    return None


# Sri Lanka trained crops (not in standard PlantVillage dataset)
SL_TRAINED_CROPS = frozenset((
    "Banana",
    "Rice",
    "Coconut",
    "Tea",
    "Chili",
    "Mango",
    "Papaya",
))

# PlantVillage crops that commonly mislabel as Sri Lanka crops (phone photo artifacts)
MISLABEL_PLANT_CROPS = frozenset((
    "Apple",
    "Blueberry",
    "Cherry_(including_sour)",
    "Corn_(maize)",
    "Grape",
    "Orange",
    "Peach",
    "Pepper,_bell",
    "Soybean",
    "Squash",
    "Raspberry",
))

# Rule-based class order when ML scores ~0 for an SL crop (proxy training)
SL_FALLBACK_DISEASES: dict[str, tuple[str, ...]] = {
    "Banana": ("Banana_Sigatoka", "Banana_Xanthomonas_wilt", "Banana_healthy"),
    "Rice": ("Rice_Blast", "Rice_Brown_spot", "Rice_Bacterial_blight", "Rice_Tungro"),
    "Tea": ("Tea_Bird_eye_spot", "Tea_Red_rust", "Tea_Anthracnose", "Tea_healthy"),
    "Chili": ("Chili_Bacterial_spot", "Chili_healthy"),
    "Papaya": ("Papaya_Ringspot", "Papaya_Anthracnose", "Papaya_Bacterial_spot", "Papaya_healthy"),
}


def crop_max_proba(classes: np.ndarray, proba: np.ndarray, crop_prefix: str) -> float:
    """Highest raw ML probability for any class in a crop family."""
    best = 0.0
    for i, name in enumerate(classes):
        if str(name).startswith(crop_prefix + "_"):
            best = max(best, float(proba[i]))
    return best


def _raw_coconut_palm_shape(metrics: dict) -> bool:
    """Palm/coconut frond shape + rust spots (no banana/mango cross-check)."""
    if _metrics_mango_oval(metrics):
        return False
    return _metrics_coconut_frond(metrics)


def is_coconut_palm_leaf(metrics: dict) -> bool:
    """
    Coconut / palm frond leaflet with rust-like spotting (not grapevine lobes).
    """
    if is_banana_sigatoka_streak_pattern(metrics) or _metrics_banana_strap(metrics):
        return False
    return _raw_coconut_palm_shape(metrics)


def is_mango_leaf(metrics: dict) -> bool:
    """
    Mango leaflet: lanceolate or oval with anthracnose-style spotting (not banana strap).
    Mango leaves have SMOOTH edges (no serration). Apple leaves have SERRATED edges.
    High serration score → Apple, NOT mango.
    """
    # ── Serration guard: serrated edges = Apple, never mango ──
    serration = float(metrics.get("serration", 0.0))
    if serration > 0.25:
        return False

    elongated = bool(metrics.get("has_elongated_lesions")) or int(
        metrics.get("elongated_patches", 0)
    ) >= 1
    if elongated and float(metrics.get("yellow_ratio", 0.0)) >= 0.12:
        if not _metrics_coconut_frond(metrics):
            return False
    lobes = int(metrics.get("lobes", 0))
    if lobes >= 3:
        return False
    aspect = float(metrics.get("aspect", 1.0))
    damage = float(metrics.get("damage_ratio", metrics.get("lesion_ratio", 0.0)))
    solidity = float(metrics.get("solidity", 1.0))
    round_spots = int(metrics.get("round_spots", 0))
    if lobes <= 2 and aspect <= 0.55:
        return True
    if lobes <= 2 and 2.0 <= aspect < 2.75:
        return True
    if (
        lobes <= 1
        and 0.95 <= aspect <= 2.65
        and 0.004 <= damage <= 0.75
        and solidity >= 0.85
        and not bool(metrics.get("has_elongated_lesions"))
        and float(metrics.get("yellow_ratio", 0.0)) < 0.20
    ):
        return True
    if round_spots >= 2 and 0.9 <= aspect <= 2.5 and lobes <= 1:
        return True
    if (
        round_spots >= 3
        and float(metrics.get("yellow_ratio", 0.0)) < 0.15
        and lobes <= 2
        and 0.35 <= aspect <= 4.5
        and not bool(metrics.get("has_elongated_lesions"))
    ):
        return True
    return False


def looks_like_mango(
    metrics: dict,
    classes: np.ndarray,
    proba: np.ndarray,
) -> bool:
    """Shape-based or ML-based mango signal (blocks Apple/Grape mislabels)."""
    if is_mango_leaf(metrics) or _metrics_mango_spotted_blade(metrics):
        return True
    mango_p = crop_max_proba(classes, proba, "Mango")
    apple_p = crop_max_proba(classes, proba, "Apple")
    grape_p = crop_max_proba(classes, proba, "Grape")

    # ── FIX 2 ── Mango threshold raised: 0.025 → 0.08, must be 1.5× above Apple.
    # Stops Apple/Banana images being overridden as Mango on weak mango signals.
    if mango_p >= 0.08 and mango_p >= apple_p * 1.5 and mango_p >= grape_p:
        return True
    if mango_p >= 0.06 and apple_p <= 0.03 and int(metrics.get("lobes", 0)) < 3:
        return True
    return False


def resolve_mango_disease(
    classes: np.ndarray,
    proba: np.ndarray,
    adjusted: np.ndarray,
    metrics: dict,
) -> tuple[int | None, float]:
    """Best mango class from ML or spot pattern (anthracnose vs healthy)."""
    idx = best_class_index_for_crop(classes, proba, adjusted, "Mango")
    if idx is not None:
        score = float(adjusted[idx]) * 0.6 + float(proba[idx]) * 0.4
        if score > 1e-12:
            return idx, max(score * 100, 48.0)

    lesion = float(metrics.get("lesion_ratio", 0.0))
    round_spots = int(metrics.get("round_spots", 0))
    if lesion >= 0.008 or round_spots >= 3:
        preferred = (
            "Mango_Anthracnose",
            "Mango_Bacterial_canker",
            "Mango_Sooty_mould",
            "Mango_healthy",
        )
    else:
        preferred = ("Mango_healthy", "Mango_Anthracnose")

    confidence = min(72.0, round(50.0 + lesion * 120, 1))
    for name in preferred:
        for i, c in enumerate(classes):
            if str(c) == name:
                return i, confidence
    return None, 0.0


def is_banana_leaf(metrics: dict) -> bool:
    """Very long banana / plantain strap — not oval mango with round spots."""
    return _metrics_banana_strap(metrics)


def is_rice_leaf(metrics: dict) -> bool:
    lobes = int(metrics.get("lobes", 0))
    aspect = float(metrics.get("aspect", 1.0))
    solidity = float(metrics.get("solidity", 1.0))
    return lobes <= 1 and aspect >= 1.8 and solidity >= 0.86


def is_tea_leaf(metrics: dict) -> bool:
    """Tea: oval leaf, fairly round, not a long mango/coconut frond."""
    lobes = int(metrics.get("lobes", 0))
    aspect = float(metrics.get("aspect", 1.0))
    solidity = float(metrics.get("solidity", 1.0))
    if (
        _metrics_mango_oval(metrics)
        or _metrics_coconut_frond(metrics)
        or _metrics_banana_strap(metrics)
        or is_banana_sigatoka_streak_pattern(metrics)
    ):
        return False
    return lobes <= 1 and 0.95 <= aspect <= 1.65 and solidity >= 0.9


def is_papaya_leaf(metrics: dict) -> bool:
    lobes = int(metrics.get("lobes", 0))
    aspect = float(metrics.get("aspect", 1.0))
    if lobes >= 3 or _metrics_coconut_frond(metrics) or _metrics_mango_oval(metrics):
        return False
    return lobes <= 2 and (aspect >= 1.5 or aspect <= 0.65)


def is_chili_leaf(metrics: dict) -> bool:
    """Chili/pepper leaflet — narrower than mango; exclude mango oval blades."""
    if _metrics_mango_oval(metrics) or is_mango_leaf(metrics):
        return False
    lobes = int(metrics.get("lobes", 0))
    aspect = float(metrics.get("aspect", 1.0))
    lesion = float(metrics.get("lesion_ratio", 0.0))
    round_spots = int(metrics.get("round_spots", 0))
    if lobes <= 1 and 1.2 <= aspect <= 2.8 and lesion >= 0.003 and round_spots <= 4:
        return True
    return False


def detect_shaped_sl_crop(metrics: dict) -> str | None:
    """Single best SL crop from leaf shape — metrics-only (no is_* cross-calls)."""
    if is_banana_sigatoka_streak_pattern(metrics) or _metrics_banana_strap(metrics):
        return "Banana"
    if _metrics_mango_oval(metrics) or is_mango_leaf(metrics):
        return "Mango"
    if _metrics_coconut_frond(metrics):
        return "Coconut"
    lobes = int(metrics.get("lobes", 0))
    aspect = float(metrics.get("aspect", 1.0))
    solidity = float(metrics.get("solidity", 1.0))
    lesion = float(metrics.get("lesion_ratio", 0.0))
    if lobes <= 1 and aspect >= 1.8 and solidity >= 0.86:
        return "Rice"
    if lobes <= 2 and (aspect >= 1.5 or aspect <= 0.65):
        return "Papaya"
    if (
        is_chili_leaf(metrics)
        and not _metrics_mango_oval(metrics)
        and not is_mango_leaf(metrics)
    ):
        return "Chili"
    if (
        lobes <= 1
        and 0.95 <= aspect <= 1.65
        and solidity >= 0.9
        and not _metrics_mango_oval(metrics)
        and not _metrics_coconut_frond(metrics)
        and not _metrics_banana_strap(metrics)
    ):
        return "Tea"
    return None


def resolve_banana_disease(
    classes: np.ndarray,
    proba: np.ndarray,
    adjusted: np.ndarray,
    metrics: dict,
) -> tuple[int | None, float]:
    """Best banana class — diseased long leaves should not be Coconut healthy."""
    idx = best_class_index_for_crop(classes, proba, adjusted, "Banana")
    if idx is not None:
        score = float(adjusted[idx]) * 0.6 + float(proba[idx]) * 0.4
        if score > 1e-12:
            return idx, max(score * 100, 48.0)

    if not (is_banana_leaf(metrics) or is_banana_sigatoka_streak_pattern(metrics)):
        return None, 0.0

    damage = float(metrics.get("damage_ratio", metrics.get("lesion_ratio", 0.0)))
    yellow = float(metrics.get("yellow_ratio", 0.0))
    if has_heavy_leaf_damage(metrics) or yellow >= 0.04 or damage >= 0.03:
        preferred = (
            "Banana_Sigatoka",
            "Banana_Xanthomonas_wilt",
            "Banana_healthy",
        )
    else:
        preferred = ("Banana_healthy", "Banana_Sigatoka")

    confidence = min(75.0, round(50.0 + max(damage, yellow) * 100, 1))
    for name in preferred:
        for i, c in enumerate(classes):
            if str(c) == name:
                return i, confidence
    return None, 0.0


def resolve_sl_disease(
    crop: str,
    classes: np.ndarray,
    proba: np.ndarray,
    adjusted: np.ndarray,
    metrics: dict,
) -> tuple[int | None, float]:
    """Best disease index for any Sri Lanka crop."""
    if crop == "Mango":
        return resolve_mango_disease(classes, proba, adjusted, metrics)
    if crop == "Banana":
        return resolve_banana_disease(classes, proba, adjusted, metrics)
    if crop == "Coconut":
        return resolve_coconut_disease(classes, proba, adjusted, metrics)

    if crop == "Papaya":
        # Prefer bacterial spot for papaya when many small round spots present
        round_spots = int(metrics.get("round_spots", 0))
        yellow = float(metrics.get("yellow_ratio", 0.0))
        if round_spots >= 2 and yellow < 0.12:
            preferred = ("Papaya_Bacterial_spot", "Papaya_Anthracnose", "Papaya_Ringspot", "Papaya_healthy")
        else:
            preferred = SL_FALLBACK_DISEASES.get("Papaya", ("Papaya_Ringspot", "Papaya_Anthracnose", "Papaya_Bacterial_spot", "Papaya_healthy"))
        lesion = float(metrics.get("lesion_ratio", 0.0))
        confidence = min(72.0, round(48.0 + lesion * 110, 1))
        for name in preferred:
            for i, c in enumerate(classes):
                if str(c) == name:
                    return i, confidence

    idx = best_class_index_for_crop(classes, proba, adjusted, crop)
    if idx is not None:
        score = float(adjusted[idx]) * 0.6 + float(proba[idx]) * 0.4
        if score > 1e-12:
            return idx, max(score * 100, 48.0)

    lesion = float(metrics.get("lesion_ratio", 0.0))
    confidence = min(72.0, round(48.0 + lesion * 110, 1))
    for name in SL_FALLBACK_DISEASES.get(crop, ()):
        for i, c in enumerate(classes):
            if str(c) == name:
                return i, confidence
    for i, c in enumerate(classes):
        if str(c).startswith(crop + "_"):
            return i, confidence
    return None, 0.0


def choose_sl_correction_target(
    metrics: dict,
    classes: np.ndarray,
    proba: np.ndarray,
    disease: str,
    raw_crop: str | None,
) -> str | None:
    """If result should be an SL crop, return that crop prefix; else None."""
    current = _class_crop(disease)
    shaped = detect_shaped_sl_crop(metrics)
    sl_crop, sl_p = best_sl_crop_from_ml(classes, proba)
    lobes = int(metrics.get("lobes", 0))

    def _pick_target() -> str | None:
        if shaped:
            return shaped
        if sl_crop and sl_p >= 0.04:
            return sl_crop
        if (
            raw_crop in SL_TRAINED_CROPS
            and crop_max_proba(classes, proba, raw_crop) >= 0.03
        ):
            return raw_crop
        return None

    target = _pick_target()
    if not target or current == target:
        if current in MISLABEL_PLANT_CROPS or current in ("Grape", "Corn_(maize)"):
            if _metrics_coconut_frond(metrics) and not _metrics_banana_strap(metrics):
                return "Coconut"
            if _metrics_mango_oval(metrics):
                return "Mango"
            if _metrics_banana_strap(metrics) or is_banana_sigatoka_streak_pattern(metrics):
                return "Banana"
            shaped_crop = detect_shaped_sl_crop(metrics)
            if shaped_crop in ("Rice", "Papaya", "Chili", "Tea"):
                return shaped_crop
        return None

    if current == "Grape" and lobes < 3:
        return target

    if current in MISLABEL_PLANT_CROPS:
        if shaped or sl_p >= 0.03:
            return target
        if crop_max_proba(classes, proba, target) >= 0.025:
            return target
        return None

    if current in SL_TRAINED_CROPS and shaped and shaped != current:
        if crop_max_proba(classes, proba, shaped) >= crop_max_proba(
            classes, proba, current
        ) * 0.8:
            return shaped

    if current == "Coconut" and "healthy" in str(disease).lower():
        if is_mango_leaf(metrics):
            return "Mango"
        if looks_like_banana_sigatoka(metrics) and has_visible_leaf_disease(metrics):
            return "Banana"
        if has_heavy_leaf_damage(metrics) and shaped == "Banana":
            return "Banana"
        if has_heavy_leaf_damage(metrics) and shaped == "Banana":
            return "Banana"

    if current == "Banana" and is_mango_leaf(metrics):
        return "Mango"

    return None


def apply_sl_crop_correction(
    metrics: dict,
    classes: np.ndarray,
    proba: np.ndarray,
    adjusted: np.ndarray,
    disease: str,
    confidence: float,
    best_idx: int,
    crop_family: str | None,
    final_crop: str | None,
    raw_crop: str | None,
) -> tuple[str, float, int, str | None, str | None, np.ndarray]:
    """
    Unified fix for all SL uploads (mango, coconut, rice, banana, tea, chili, papaya).
    Stops Apple/Grape/Corn 45% wrong labels on Sri Lanka leaf photos.
    """
    banana_ml = crop_max_proba(classes, proba, "Banana")
    mango_ml = crop_max_proba(classes, proba, "Mango")
    mango_spotted = _metrics_mango_spotted_blade(metrics)

    try:
        if is_banana_leaf(metrics):
            papaya_ml = crop_max_proba(classes, proba, "Papaya")
            banana_ml = crop_max_proba(classes, proba, "Banana")
            shaped = detect_shaped_sl_crop(metrics)
            if papaya_ml >= 0.03 and papaya_ml > banana_ml * 1.2 and shaped != "Papaya":
                adjusted = adjusted.copy()
                for i, name in enumerate(classes):
                    n = str(name)
                    if n.startswith("Banana_"):
                        adjusted[i] = max(float(adjusted[i]), 0.35)
                    if n.startswith("Papaya_"):
                        adjusted[i] = float(adjusted[i]) * 0.5
                best_idx = int(np.argmax(adjusted))
                return (
                    str(classes[best_idx]),
                    round(max(float(adjusted[best_idx]) * 100.0, 48.0), 1),
                    best_idx,
                    "Banana",
                    "Banana",
                    adjusted,
                )
    except Exception:
        pass

    try:
        if str(disease).startswith("Papaya_") and is_banana_leaf(metrics):
            b_idx, b_conf = resolve_banana_disease(classes, proba, adjusted, metrics)
            if b_idx is not None:
                adjusted = adjusted.copy()
                adjusted[b_idx] = max(float(adjusted[b_idx]), 0.35)
                return (
                    str(classes[b_idx]),
                    round(max(b_conf, 48.0), 1),
                    b_idx,
                    "Banana",
                    "Banana",
                    adjusted,
                )
            adjusted = adjusted.copy()
            for i, name in enumerate(classes):
                n = str(name)
                if n.startswith("Banana_"):
                    adjusted[i] = max(float(adjusted[i]), 0.35)
                if n.startswith("Papaya_"):
                    adjusted[i] = float(adjusted[i]) * 0.5
            best_idx = int(np.argmax(adjusted))
            if str(classes[best_idx]).startswith("Banana_"):
                return (
                    str(classes[best_idx]),
                    round(max(float(adjusted[best_idx]) * 100.0, 48.0), 1),
                    best_idx,
                    "Banana",
                    "Banana",
                    adjusted,
                )
    except Exception:
        pass

    if mango_spotted and not is_banana_sigatoka_streak_pattern(metrics):
        if _class_crop(disease) == "Banana" or banana_ml <= max(mango_ml * 2.5, 0.08):
            m_idx, m_conf = resolve_mango_disease(classes, proba, adjusted, metrics)
            if m_idx is not None:
                adjusted = adjusted.copy()
                adjusted[m_idx] = max(float(adjusted[m_idx]), 0.38)
                return (
                    str(classes[m_idx]),
                    round(max(m_conf, 55.0), 1),
                    m_idx,
                    "Mango",
                    "Mango",
                    adjusted,
                )

    if (
        "healthy" in str(disease).lower()
        and has_visible_leaf_disease(metrics)
        and float(metrics.get("lesion_ratio", 0.0)) >= 0.08
        and is_confirmed_banana_leaf(metrics)
    ):
        b_idx, b_conf = resolve_banana_disease(classes, proba, adjusted, metrics)
        if b_idx is not None:
            adjusted = adjusted.copy()
            adjusted[b_idx] = max(float(adjusted[b_idx]), 0.35)
            return (
                str(classes[b_idx]),
                round(max(b_conf, 48.0), 1),
                b_idx,
                "Banana",
                "Banana",
                adjusted,
            )

    if (
        str(disease) == "Coconut_healthy"
        and is_confirmed_banana_leaf(metrics)
        and has_visible_leaf_disease(metrics)
    ):
        b_idx, b_conf = resolve_banana_disease(classes, proba, adjusted, metrics)
        if b_idx is not None:
            adjusted = adjusted.copy()
            adjusted[b_idx] = max(float(adjusted[b_idx]), 0.42)
            return (
                str(classes[b_idx]),
                round(b_conf, 1),
                b_idx,
                "Banana",
                "Banana",
                adjusted,
            )

    if (
        _raw_coconut_palm_shape(metrics)
        and not is_confirmed_banana_leaf(metrics)
        and not _metrics_mango_oval(metrics)
    ):
        idx, conf = resolve_coconut_disease(classes, proba, adjusted, metrics)
        if idx is not None:
            adjusted = adjusted.copy()
            adjusted[idx] = max(float(adjusted[idx]), 0.35)
            return (
                str(classes[idx]),
                round(conf, 1),
                idx,
                "Coconut",
                "Coconut",
                adjusted,
            )

    chili_ml = crop_max_proba(classes, proba, "Chili")
    mango_shape = (
        mango_spotted
        or _metrics_mango_oval(metrics)
        or is_mango_leaf(metrics)
        or looks_like_mango(metrics, classes, proba)
    )
    if mango_shape and not is_banana_sigatoka_streak_pattern(metrics):
        if (
            mango_ml >= chili_ml * 0.5
            or mango_ml >= 0.015
            or _class_crop(disease) in ("Chili", "Banana", "Tea")
        ):
            m_idx, m_conf = resolve_mango_disease(classes, proba, adjusted, metrics)
            if m_idx is not None:
                adjusted = adjusted.copy()
                adjusted[m_idx] = max(float(adjusted[m_idx]), 0.38)
                return (
                    str(classes[m_idx]),
                    round(max(m_conf, 52.0), 1),
                    m_idx,
                    "Mango",
                    "Mango",
                    adjusted,
                )

    if (
        detect_shaped_sl_crop(metrics) == "Chili"
        and not mango_shape
        and chili_ml >= max(0.02, mango_ml * 1.2)
    ):
        idx, conf = resolve_sl_disease("Chili", classes, proba, adjusted, metrics)
        if idx is not None:
            adjusted = adjusted.copy()
            adjusted[idx] = max(float(adjusted[idx]), 0.42)
            return (
                str(classes[idx]),
                round(conf, 1),
                idx,
                "Chili",
                "Chili",
                adjusted,
            )

    target = choose_sl_correction_target(metrics, classes, proba, disease, raw_crop)
    if not target:
        return disease, confidence, best_idx, crop_family, final_crop, adjusted

    idx, conf = resolve_sl_disease(target, classes, proba, adjusted, metrics)
    if idx is None:
        return disease, confidence, best_idx, crop_family, final_crop, adjusted

    adjusted = adjusted.copy()
    adjusted[idx] = max(float(adjusted[idx]), 0.42)
    out_disease = str(classes[idx])
    out_conf = round(conf, 1)
    out_crop = target

    if (
        out_disease.endswith("_healthy")
        and is_confirmed_banana_leaf(metrics)
        and has_visible_leaf_disease(metrics)
    ):
        b_idx, b_conf = resolve_banana_disease(classes, proba, adjusted, metrics)
        if b_idx is not None:
            adjusted[b_idx] = max(float(adjusted[b_idx]), 0.42)
            return (
                str(classes[b_idx]),
                round(b_conf, 1),
                b_idx,
                "Banana",
                "Banana",
                adjusted,
            )

    if out_crop == "Banana" and (
        is_mango_leaf(metrics) or _metrics_mango_spotted_blade(metrics)
    ):
        if not is_banana_sigatoka_streak_pattern(metrics):
            m_idx, m_conf = resolve_mango_disease(classes, proba, adjusted, metrics)
            if m_idx is not None:
                adjusted[m_idx] = max(float(adjusted[m_idx]), 0.42)
                return (
                    str(classes[m_idx]),
                    round(max(m_conf, 55.0), 1),
                    m_idx,
                    "Mango",
                    "Mango",
                    adjusted,
                )

    return (
        out_disease,
        out_conf,
        idx,
        out_crop,
        out_crop,
        adjusted,
    )


def should_block_plantvillage_override(
    metrics: dict,
    classes: np.ndarray,
    proba: np.ndarray,
    crop_family: str | None,
) -> bool:
    """Block the old 45% vision override when an SL crop is likely."""
    if detect_shaped_sl_crop(metrics):
        return True
    sl_crop, sl_p = best_sl_crop_from_ml(classes, proba)
    if sl_p >= 0.03 and crop_family not in SL_TRAINED_CROPS:
        return True
    if crop_family in MISLABEL_PLANT_CROPS and sl_p >= 0.02:
        return True
    return False


def best_sl_crop_from_ml(classes: np.ndarray, proba: np.ndarray) -> tuple[str | None, float]:
    """Sri Lanka crop family with highest raw ML probability."""
    best_crop = None
    best_p = 0.0
    for crop in SL_TRAINED_CROPS:
        p = crop_max_proba(classes, proba, crop)
        if p > best_p:
            best_p, best_crop = p, crop
    return best_crop, best_p


def is_likely_a_leaf(metrics: dict) -> tuple[bool, str]:
    """
    Check if shape metrics suggest the image is actually a plant leaf.
    Returns (is_leaf: bool, reason: str).

    Rejects:
    - Extreme aspect ratio (too wide or too tall for any known leaf)
    - Very low solidity (too irregular/porous — likely non-organic)
    - Very high solidity + small area (likely a manufactured object)
    - No green detected (mask empty)
    - Too circular / compact (manufactured objects)
    - Very low green ratio (mostly not green)
    - Too few contour points (shape too simple)
    """
    aspect = float(metrics.get("aspect", 1.0))
    solidity = float(metrics.get("solidity", 1.0))
    area = float(metrics.get("area", 0))
    has_green = bool(metrics.get("has_green", True))
    green_ratio = float(metrics.get("green_ratio", 1.0))
    compactness = float(metrics.get("compactness", 0))
    contour_points = int(metrics.get("contour_points", 0))
    extent = float(metrics.get("extent", 1.0))
    num_regions = int(metrics.get("num_regions", 0))
    border_touching = bool(metrics.get("border_touching", False))
    # Coverage: fraction of image occupied by the largest green region
    coverage = area / 65536.0  # 256*256 resized image

    # Must have a contour
    if area <= 0:
        return False, "no leaf contour detected"

    # Border + many regions → complex scene with scattered greenery, not a single leaf
    if border_touching and num_regions > 3:
        return False, f"green touches border with {num_regions} regions — not a single leaf"

    # Too many distinct green regions suggests complex scene, not a single leaf
    if num_regions > 5:
        return False, f"too many green regions ({num_regions}) — not a single leaf"

    # Too little green → likely small background patch, not a leaf worth diagnosing
    if coverage < 0.10:
        return False, f"green coverage too small ({coverage:.0%}) — leaf too small or background"

    # If green covers too much of the image (>80%), it's likely wall/ground not a leaf
    if coverage > 0.80:
        return False, f"green coverage too large ({coverage:.0%}) — not an isolated leaf"

    # Low solidity — too porous/irregular for a real leaf (real leaves: 0.70-0.95)
    if solidity < 0.55:
        return False, f"solidity too low ({solidity:.2f} < 0.55)"

    # Very high solidity + low green → manufactured object (shoe, tool, etc.)
    if solidity > 0.97 and green_ratio < 0.30:
        return False, f"high solidity ({solidity:.2f}) + low green ({green_ratio:.2f}) — not a leaf"

    # Extreme aspect ratios
    if aspect > 4.0:
        return False, f"aspect ratio too extreme ({aspect:.2f} > 4.0)"
    if aspect < 0.25:
        return False, f"aspect ratio too extreme ({aspect:.2f} < 0.25)"

    # Too circular/compact (compactness < 1.2 = very circular, like a coin, button, lens)
    if compactness > 0 and compactness < 1.2:
        return False, f"too circular (compactness={compactness:.2f}) — not a leaf"

    # Too rectangular (extent > 0.85 = fills bounding box, like a phone, book)
    if extent > 0.88:
        return False, f"too rectangular (extent={extent:.2f}) — not a leaf"

    # Very simple shape (too few contour points → manufactured object)
    if contour_points > 0 and contour_points < 20:
        return False, f"shape too simple ({contour_points} points) — not a leaf"

    # No green at all
    if not has_green:
        return False, "no green pixels detected"

    # Very low green percentage — unlikely a leaf
    if green_ratio > 0 and green_ratio < 0.05:
        return False, f"green ratio too low ({green_ratio:.4f}) — not a leaf"

    return True, ""


def should_apply_coconut_override(
    metrics: dict,
    classes: np.ndarray,
    proba: np.ndarray,
    disease: str,
    raw_crop: str | None,
) -> bool:
    """
    Only fix Grape/Corn mislabels on real coconut-style fronds.
    Do not override when ML clearly prefers another Sri Lanka crop.
    """
    if _metrics_mango_oval(metrics):
        return False
    if not _metrics_coconut_frond(metrics):
        return False
    if crop_max_proba(classes, proba, "Mango") >= crop_max_proba(classes, proba, "Coconut"):
        return False

    final_crop = _class_crop(disease)
    if final_crop in ("Grape", "Corn_(maize)"):
        return True

    coco = crop_max_proba(classes, proba, "Coconut")
    if coco >= 0.03:
        return True

    if raw_crop and raw_crop != "Coconut":
        other = crop_max_proba(classes, proba, raw_crop)
        if other >= max(coco * 2.5, 0.10):
            return False

    return final_crop == "Coconut" and coco < 0.03


def resolve_coconut_disease(
    classes: np.ndarray,
    proba: np.ndarray,
    adjusted: np.ndarray,
    metrics: dict,
) -> tuple[int | None, float]:
    """
    Pick best coconut class. Uses ML when non-zero; otherwise rule-based
    (model often scores 0 on coconut because training used papaya proxies).
    """
    idx = best_class_index_for_crop(classes, proba, adjusted, "Coconut")
    if idx is not None:
        score = float(adjusted[idx]) * 0.6 + float(proba[idx]) * 0.4
        if score > 1e-12:
            return idx, max(score * 100, 48.0)

    if is_banana_leaf(metrics):
        return None, 0.0

    lesion_ratio = float(metrics.get("lesion_ratio", 0.0))
    if looks_like_banana_sigatoka(metrics):
        return None, 0.0

    if has_heavy_leaf_damage(metrics):
        preferred = ("Coconut_Gray_leaf_spot", "Coconut_Leaf_rot")
    elif lesion_ratio >= 0.08:
        preferred = ("Coconut_Gray_leaf_spot", "Coconut_Leaf_rot", "Coconut_healthy")
    elif lesion_ratio >= 0.03:
        preferred = ("Coconut_Leaf_rot", "Coconut_Gray_leaf_spot", "Coconut_healthy")
    else:
        preferred = ("Coconut_healthy", "Coconut_Leaf_rot")

    confidence = min(75.0, round(50.0 + lesion_ratio * 100, 1))
    for name in preferred:
        for i, c in enumerate(classes):
            if str(c) == name:
                return i, confidence
    return None, 0.0


def best_class_index_for_crop(
    classes: np.ndarray,
    proba: np.ndarray,
    adjusted: np.ndarray,
    crop_prefix: str,
) -> int | None:
    """Highest-probability class index within one crop family."""
    best_i = None
    best_score = -1.0
    for i, name in enumerate(classes):
        if str(name).startswith(crop_prefix + "_"):
            score = float(adjusted[i]) * 0.6 + float(proba[i]) * 0.4
            if score > best_score:
                best_score = score
                best_i = i
    return best_i


PREDICT_RULES_VERSION = "sl-unified-v14-mango-not-banana"


def finalize_sl_prediction(
    metrics: dict,
    classes: np.ndarray,
    proba: np.ndarray,
    adjusted: np.ndarray,
    disease: str,
    confidence: float,
    best_idx: int,
    crop_family: str | None,
    final_crop: str | None,
) -> tuple[str, float, int, str | None, str | None, np.ndarray]:
    """
    Last pass: only fix Coconut_healthy on clearly banana-shaped diseased leaves.
    Does not override rice/mango/tea/etc. disease labels.
    """
    # Chili / Banana / Tea label on clear mango leaf (common phone-upload mistake)
    if _class_crop(disease) in ("Chili", "Banana", "Tea") and (
        is_mango_leaf(metrics)
        or _metrics_mango_oval(metrics)
        or _metrics_mango_spotted_blade(metrics)
        or looks_like_mango(metrics, classes, proba)
    ):
        if is_banana_sigatoka_streak_pattern(metrics):
            return disease, confidence, best_idx, crop_family, final_crop, adjusted

        # ── FIX 3 ── Chili→Mango override multiplier raised: 0.25 → 0.60
        # Mango must score at least 60% of Chili score before overriding.
        # Prevents Chili/Banana/Tea results being wrongly flipped to Mango
        # when the mango ML signal is only marginally present.
        if crop_max_proba(classes, proba, "Mango") >= crop_max_proba(classes, proba, "Chili") * 0.60:
            m_idx, m_conf = resolve_mango_disease(classes, proba, adjusted, metrics)
            if m_idx is not None:
                adjusted = adjusted.copy()
                adjusted[m_idx] = max(float(adjusted[m_idx]), 0.5)
                return (
                    str(classes[m_idx]),
                    round(max(m_conf, 55.0), 1),
                    m_idx,
                    "Mango",
                    "Mango",
                    adjusted,
                )

    if str(disease) != "Coconut_healthy":
        return disease, confidence, best_idx, crop_family, final_crop, adjusted

    if not has_visible_leaf_disease(metrics):
        return disease, confidence, best_idx, crop_family, final_crop, adjusted

    banana_confirmed = is_confirmed_banana_leaf(metrics)
    if not banana_confirmed:
        banana_p = crop_max_proba(classes, proba, "Banana")
        coconut_p = crop_max_proba(classes, proba, "Coconut")
        banana_confirmed = banana_p >= 0.08 and banana_p >= coconut_p * 1.5

    if banana_confirmed:
        b_idx, b_conf = resolve_banana_disease(classes, proba, adjusted, metrics)
        if b_idx is not None:
            adjusted = adjusted.copy()
            adjusted[b_idx] = max(float(adjusted[b_idx]), 0.48)
            return (
                str(classes[b_idx]),
                round(max(b_conf, 50.0), 1),
                b_idx,
                "Banana",
                "Banana",
                adjusted,
            )

    for name in ("Coconut_Gray_leaf_spot", "Coconut_Leaf_rot"):
        for i, c in enumerate(classes):
            if str(c) == name:
                adjusted = adjusted.copy()
                adjusted[i] = max(float(adjusted[i]), 0.45)
                return (
                    str(classes[i]),
                    round(max(50.0, confidence), 1),
                    i,
                    "Coconut",
                    "Coconut",
                    adjusted,
                )

    return disease, confidence, best_idx, crop_family, final_crop, adjusted


def map_disease(plant_type: str, metrics: dict) -> tuple[str, float]:
    """
    Generic OpenCV disease mapper using lesion and shape metrics.
    Returns (disease_name, confidence) for any supported crop.
    Confidence 0-100 reflects how strongly the visual signs match.
    """
    lesion = float(metrics.get("lesion_ratio", 0.0))
    yellow = float(metrics.get("yellow_ratio", 0.0))
    damage = float(metrics.get("damage_ratio", lesion))
    round_spots = int(metrics.get("round_spots", 0))
    elongated = int(metrics.get("elongated_patches", 0))

    # Healthy baseline — very low damage
    if damage < 0.02 and yellow < 0.02 and round_spots == 0:
        return "healthy", 75.0

    pt = plant_type.lower().strip()

    # ── Apple ──
    if pt == "apple":
        if yellow > 0.08 and round_spots >= 1:
            return "Cedar_apple_rust", min(85.0, 50.0 + yellow * 300)
        if round_spots >= 2 and lesion >= 0.05:
            return "Black_rot", min(80.0, 50.0 + lesion * 200)
        if lesion >= 0.02 or round_spots >= 1:
            return "Apple_scab", min(75.0, 45.0 + lesion * 200)
        return "healthy", 60.0

    # ── Banana ──
    if pt == "banana":
        streaks = int(metrics.get("elongated_patches", 0))
        if streaks >= 2 or (damage > 0.04 and yellow > 0.03):
            return "Sigatoka", min(80.0, 50.0 + damage * 200)
        if damage > 0.06 and lesion > 0.04:
            return "Xanthomonas_wilt", min(75.0, 45.0 + damage * 150)
        return "healthy", 60.0

    # ── Mango ──
    if pt == "mango":
        if round_spots >= 3:
            return "Anthracnose", min(80.0, 50.0 + lesion * 200)
        if damage > 0.04 and lesion > 0.02:
            return "Bacterial_canker", min(72.0, 45.0 + lesion * 180)
        if damage < 0.01 and round_spots == 0:
            return "healthy", 70.0
        return "Anthracnose", 55.0

    # ── Rice ──
    if pt == "rice":
        if elongated >= 2 or (damage > 0.04 and lesion > 0.02):
            return "Blast", min(78.0, 50.0 + damage * 180)
        if yellow > 0.05:
            return "Bacterial_blight", min(72.0, 45.0 + yellow * 200)
        if round_spots >= 2 and lesion > 0.03:
            return "Brown_spot", min(70.0, 45.0 + lesion * 180)
        return "healthy", 60.0

    # ── Tea ──
    if pt == "tea":
        if yellow > 0.06 and damage > 0.03:
            return "Red_rust", min(75.0, 45.0 + yellow * 200)
        if round_spots >= 2:
            return "Bird_eye_spot", min(72.0, 45.0 + round_spots * 8)
        if damage > 0.05:
            return "Brown_blight", min(68.0, 40.0 + damage * 150)
        if lesion > 0.03:
            return "Anthracnose", min(65.0, 40.0 + lesion * 150)
        return "healthy", 60.0

    # ── Papaya ──
    if pt == "papaya":
        if round_spots >= 2 and yellow < 0.12:
            return "Bacterial_spot", min(75.0, 50.0 + round_spots * 8)
        if damage > 0.05 and round_spots >= 1:
            return "Anthracnose", min(72.0, 45.0 + lesion * 180)
        if yellow > 0.04 or damage > 0.04:
            return "Ringspot", min(68.0, 40.0 + damage * 150)
        return "healthy", 60.0

    # ── Chili ──
    if pt == "chili":
        if round_spots >= 1 or lesion > 0.02:
            return "Bacterial_spot", min(75.0, 50.0 + lesion * 200)
        return "healthy", 65.0

    # ── Tomato ──
    if pt == "tomato":
        if lesion > 0.05 and damage > 0.06 and round_spots >= 2:
            return "Early_blight", min(80.0, 50.0 + lesion * 180)
        if damage > 0.05 and lesion > 0.03 and yellow > 0.03:
            return "Late_blight", min(75.0, 45.0 + damage * 150)
        if round_spots >= 3 and lesion > 0.03:
            return "Septoria_leaf_spot", min(72.0, 45.0 + round_spots * 6)
        if round_spots >= 1 or lesion > 0.02:
            return "Bacterial_spot", min(65.0, 40.0 + lesion * 200)
        return "healthy", 60.0

    # ── Potato ──
    if pt == "potato":
        if lesion > 0.05 and round_spots >= 2:
            return "Early_blight", min(78.0, 50.0 + lesion * 180)
        if damage > 0.05 and lesion > 0.03:
            return "Late_blight", min(72.0, 45.0 + damage * 150)
        return "healthy", 60.0

    # ── Corn ──
    if pt == "corn" or pt == "corn_(maize)":
        if elongated >= 2 or (damage > 0.04 and lesion > 0.03):
            return "Northern_Leaf_Blight", min(75.0, 50.0 + damage * 160)
        if round_spots >= 2 and damage > 0.03:
            return "Cercospora_leaf_spot_Gray_leaf_spot", min(70.0, 45.0 + round_spots * 7)
        return "healthy", 60.0

    # ── Grape ──
    if pt == "grape":
        if round_spots >= 2 and lesion > 0.04:
            return "Black_rot", min(78.0, 50.0 + lesion * 180)
        if damage > 0.04 and lesion > 0.02:
            return "Leaf_blight_(Isariopsis_Leaf_Spot)", min(68.0, 40.0 + damage * 150)
        return "healthy", 60.0

    # ── Coconut ──
    if pt == "coconut":
        if damage > 0.04 and lesion > 0.02:
            return "Leaf_rot", min(72.0, 45.0 + damage * 180)
        if round_spots >= 1:
            return "Gray_leaf_spot", min(68.0, 45.0 + round_spots * 8)
        return "healthy", 60.0

    # ── Default: not enough visual heuristics for this crop ──
    if damage > 0.03:
        return "unknown", 40.0
    return "healthy", 55.0


def format_display_name(class_name: str) -> str:
    """Human-readable plant + disease label."""
    name = str(class_name)
    if name.endswith("_healthy") or name.endswith("healthy"):
        plant = name.replace("_healthy", "").replace("___", " ").replace("__", " ").replace("_", " ")
        return f"{plant.strip()} - Healthy"
    parts = name.split("_", 1)
    if len(parts) == 2:
        plant = parts[0].replace("___", " ").replace("__", " ").replace(",", " ")
        disease = parts[1].replace("___", " ").replace("__", " ").replace("_", " ")
        return f"{plant.strip()} - {disease.strip()}"
    return name.replace("_", " ")