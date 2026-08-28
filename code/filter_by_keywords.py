from pathlib import Path
from datetime import datetime, timezone
import json

input_file = Path("all_videos_2024-01_to_2025-10.json")
output_file = Path("all_filtered_videos_by_keywords.json")

# 把你要找的關鍵字放在這裡
keywords = [
    "罷免",
    "二階罷免",
    "大罷免",
]

matched_videos = []

with open(input_file, "r", encoding="utf-8") as f:
    payload = json.load(f)

videos = payload["data"] if isinstance(payload, dict) and "data" in payload else payload

for video in videos:
    title = video.get("snippet", {}).get("title", "")
    description = video.get("snippet", {}).get("description", "")
    text = f"{title}\n{description}"

    matched_keywords = [keyword for keyword in keywords if keyword in text]

    if matched_keywords:
        video_copy = dict(video)
        video_copy["matched_keywords"] = matched_keywords
        matched_videos.append(video_copy)

result = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "generated_by": "filter_by_keywords.py",
    "statistics": {
        "input_file": str(input_file),
        "input_video_count": len(videos),
        "matched_video_count": len(matched_videos),
        "keywords": keywords,
    },
    "data": matched_videos,
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("原始影片數:", len(videos))
print("符合關鍵字的影片數:", len(matched_videos))
print("輸出檔案:", output_file)
