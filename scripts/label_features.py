"""Label each alive SAE feature from a montage of its top-activating images.

A multimodal LLM (claude-opus-4-6) names the shared visual concept; bounded concurrency
with a resumable JSONL checkpoint. Run per SAE: --which cls|patch. The produced labels are
what the dashboard reads from results/feature_labels/ to name clusters.
"""
import os
import sys
import json
import argparse
import asyncio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import autointerp_common as ai  # noqa: E402

SYSTEM_CLS = (
    "You are analyzing one feature of a sparse autoencoder trained on DINOv2 image "
    "embeddings. You are shown a montage of the images that most strongly activate this "
    "feature. Identify the SINGLE visual concept these images share. The concept may be an "
    "object, animal, material, texture, color, scene type, or abstract visual pattern. Be "
    "specific but general enough to cover the examples. If the images share no clear concept, "
    "say so and mark low confidence."
)
SYSTEM_PATCH = (
    "You are analyzing one feature of a sparse autoencoder trained on DINOv2 patch-token "
    "embeddings. Each montage tile is an image with a translucent RED heatmap marking the "
    "image region (patches) where this feature fires most. Focus on what is under the red "
    "highlight. Identify the SINGLE visual concept the highlighted regions share (object part, "
    "texture, material, color, shape, etc.). Be specific but general enough to cover the examples."
)
INSTR = (
    '\n\nReturn STRICT JSON only: {"label": "<3-7 word concept>", '
    '"confidence": <0-1>, "polysemantic": <true|false>, "notes": "<one sentence>"}'
)


def make_worker(client, system, model):
    async def worker(feat):
        content = [ai.img_block(feat["label_montage"]),
                   {"type": "text",
                    "text": f"These {len(feat['label_gidx'])} tiles maximally activate the "
                            f"feature." + INSTR}]
        txt = await ai.call(client, system, content, model=model, max_tokens=300)
        try:
            parsed = ai.parse_json(txt)
        except Exception:
            parsed = {"label": "PARSE_ERROR", "confidence": 0.0, "polysemantic": None,
                      "notes": txt[:200]}
        return {"feature": feat["feature"], "which": feat["which"], **parsed}
    return worker


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["cls", "patch"], required=True)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--model", default=ai.MODEL)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--limit", type=int, default=0, help="debug: only N features")
    args = ap.parse_args()
    ai.load_env()
    print(f"[label {args.which}] model={args.model} results_dir={args.results_dir}", flush=True)

    catalog = json.load(open(os.path.join(args.data_dir, f"catalog_{args.which}.json")))
    if args.limit:
        catalog = catalog[: args.limit]
    system = SYSTEM_CLS if args.which == "cls" else SYSTEM_PATCH
    client = ai.get_client()
    out = os.path.join(args.results_dir, f"labels_{args.which}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    await ai.run_all(catalog, make_worker(client, system, args.model), out, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
