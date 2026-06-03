"""Shared helpers for multimodal auto-interp: async Anthropic client, image encoding,
robust JSON parsing, concurrency + JSONL checkpointing."""
import os
import io
import json
import base64
import asyncio
import traceback
from PIL import Image

MODEL = "claude-opus-4-6"


def load_env(path="/workspace-vast/nickj/projects/.env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_client():
    from anthropic import AsyncAnthropic
    return AsyncAnthropic(max_retries=8)


def img_block(path, max_px=900):
    """Return an Anthropic image content block (base64 JPEG), downscaled if huge."""
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_px:
        s = max_px / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def parse_json(text):
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        t = t[a:b + 1]
    return json.loads(t)


async def call(client, system, content, model=MODEL, max_tokens=400, retries=10):
    """One messages.create with manual backoff on top of SDK retries.
    Concurrency is bounded by the caller (run_all's semaphore). Handles transient
    429/500/529 overload by retrying with capped exponential backoff."""
    for attempt in range(retries):
        try:
            resp = await client.messages.create(
                model=model, max_tokens=max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(min(90, 3 * 2 ** attempt))
    return None


def load_done(path, key="feature"):
    """Features with at least one SUCCESSFUL record (no 'error', not a PARSE_ERROR label).
    Error records are not counted, so they get retried on resume."""
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                if "error" in r:
                    continue
                if r.get("label") == "PARSE_ERROR":
                    continue
                done.add(r[key])
            except Exception:
                pass
    return done


async def run_all(items, worker, out_path, concurrency=100, key="feature"):
    """Run `worker(item)->dict` over items with bounded concurrency, append JSONL,
    skipping items whose key is already present in out_path."""
    done = load_done(out_path, key)
    todo = [it for it in items if it[key] not in done]
    print(f"[run_all] {len(todo)} to do, {len(done)} already done -> {out_path}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    f = open(out_path, "a")
    n_ok = [0]
    n_err = [0]

    async def one(it):
        async with sem:
            try:
                rec = await worker(it)
            except Exception as e:
                n_err[0] += 1
                rec = {key: it[key], "error": str(e)[:300]}
            async with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_ok[0] += 1
                if n_ok[0] % 200 == 0:
                    print(f"[run_all] {n_ok[0]}/{len(todo)} (err={n_err[0]})", flush=True)

    # note: worker acquires nothing; semaphore here bounds total concurrency
    await asyncio.gather(*[one(it) for it in todo])
    f.close()
    print(f"[run_all] complete ok={n_ok[0]} err={n_err[0]}", flush=True)
