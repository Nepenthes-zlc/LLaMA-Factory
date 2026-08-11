"""Convert ms-swift training data to LLaMA-Factory format.

Adds <image> tag before 'See the attached image.' in user content,
and includes system message in the messages list.
"""
import json
import sys

src = sys.argv[1]
dst = sys.argv[2]

data = json.load(open(src))
out = []

for sample in data:
    msgs = []
    for m in sample["messages"]:
        msg = {"role": m["role"], "content": m["content"]}
        if m["role"] == "user" and sample.get("images"):
            c = msg["content"]
            if "See the attached image." in c:
                c = c.replace("See the attached image.", "<image>See the attached image.", 1)
            elif "The initial screenshot (Step 0) is attached as the image." in c:
                c = c.replace("The initial screenshot (Step 0) is attached as the image.",
                              "<image>The initial screenshot (Step 0) is attached as the image.", 1)
            msg["content"] = c
        msgs.append(msg)

    out.append({
        "messages": msgs,
        "images": sample.get("images", []),
    })

with open(dst, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Converted {len(out)} samples -> {dst}")
