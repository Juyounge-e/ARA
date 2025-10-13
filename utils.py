import json

def parse_results(results, model_names):
    objects = []
    for *box, conf, cls in results.xyxy[0].tolist():
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        label = model_names[int(cls)]

        objects.append({
            "label": label,
            "confidence": round(conf, 2),
            "center": {"x": int(cx), "y": int(cy)}
        })
    return json.dumps(objects, indent=2)