"""Generate synthetic acupuncture RWD."""
import json, random, os
random.seed(42)
acupoints = ["BL23","BL25","BL40","GV3","GV4","KI3","SP6","LI4","ST36"]

def gen_patient(pid):
    return {"patient_id": f"A{pid}", "acupoints": random.sample(acupoints, random.randint(3,6)),
            "sessions": random.randint(5,20), "pain_before": random.uniform(5,9),
            "pain_after": random.uniform(1,7), "outcome": random.choice(["improved","stable"])}

if __name__ == "__main__":
    data = [gen_patient(i) for i in range(1000)]
    out = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_acupuncture.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(data)} patients -> {out}")
