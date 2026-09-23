"""Local playground: prepare data, inspect language tags, copy a train recipe."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from indictrans2_finetune.data import prepare_file
from indictrans2_finetune.langs import LANGUAGES, default_model, normalize_lang

SAMPLE_EN = (
    "Your Aadhaar enrolment is complete. You can download the e-Aadhaar from the UIDAI portal."
)

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IndicTrans2 Fine-tune</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #0f1419; color: #e7ecf3; }
    main { max-width: 920px; margin: 0 auto; padding: 32px 20px 64px; }
    h1 { font-size: 1.75rem; margin: 0 0 8px; }
    p.lead { color: #b4c0ce; line-height: 1.5; }
    nav { display: flex; gap: 8px; flex-wrap: wrap; margin: 24px 0; }
    nav button { background: #1c2733; color: #e7ecf3; border: 1px solid #334155; border-radius: 999px; padding: 8px 14px; cursor: pointer; }
    nav button.active { background: #2563eb; border-color: #2563eb; }
    section { display: none; background: #151c24; border: 1px solid #243040; border-radius: 16px; padding: 20px; }
    section.visible { display: block; }
    label { display: block; font-size: 0.85rem; color: #9fb0c3; margin: 12px 0 6px; }
    select, input, textarea { width: 100%; box-sizing: border-box; background: #0f1419; color: #e7ecf3; border: 1px solid #334155; border-radius: 10px; padding: 10px 12px; }
    textarea { min-height: 90px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 700px) { .row { grid-template-columns: 1fr; } }
    button.primary { margin-top: 16px; background: #2563eb; color: white; border: 0; border-radius: 10px; padding: 10px 16px; cursor: pointer; font-weight: 600; }
    pre { background: #0b1016; border-radius: 12px; padding: 14px; overflow: auto; font-size: 0.85rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #243040; }
    .err { color: #fca5a5; }
  </style>
</head>
<body>
  <main>
    <h1>IndicTrans2 LoRA fine-tune</h1>
    <p class="lead">Prepare parallel data in AI4Bharat’s official pair-folder layout, pick the matching
      en-indic / indic-en / indic-indic checkpoint, and LoRA-adapt it to your domain.</p>
    <nav>
      <button class="active" data-tab="recipe">Recipe</button>
      <button data-tab="prepare">Prepare data</button>
      <button data-tab="translate">Translate</button>
      <button data-tab="langs">Languages</button>
    </nav>
    <section id="recipe" class="visible">
      <div class="row">
        <div>
          <label for="src">Source</label>
          <select id="src"></select>
        </div>
        <div>
          <label for="tgt">Target</label>
          <select id="tgt"></select>
        </div>
      </div>
      <div class="row">
        <div>
          <label for="size">Checkpoint</label>
          <select id="size">
            <option value="distilled">distilled (faster, recommended to start)</option>
            <option value="base">base 1B / 320M</option>
          </select>
        </div>
        <div>
          <label for="steps">Max steps</label>
          <input id="steps" type="number" value="400" min="50" step="50" />
        </div>
      </div>
      <pre id="recipe-out"></pre>
    </section>
    <section id="prepare">
      <p class="lead">JSONL with src/tgt (or FLORES keys) or CSV with a header. Language columns can be omitted if you set the dropdowns.</p>
      <div class="row">
        <div>
          <label>Train file</label>
          <input id="train-file" type="file" accept=".jsonl,.csv,.tsv,.json" />
        </div>
        <div>
          <label>Dev file</label>
          <input id="dev-file" type="file" accept=".jsonl,.csv,.tsv,.json" />
        </div>
      </div>
      <div class="row">
        <div>
          <label for="p-src">Default source</label>
          <select id="p-src"></select>
        </div>
        <div>
          <label for="p-tgt">Default target</label>
          <select id="p-tgt"></select>
        </div>
      </div>
      <button class="primary" id="convert">Convert to official layout</button>
      <pre id="prepare-out"></pre>
    </section>
    <section id="translate">
      <div class="row">
        <div>
          <label for="t-src">Source</label>
          <select id="t-src"></select>
        </div>
        <div>
          <label for="t-tgt">Target</label>
          <select id="t-tgt"></select>
        </div>
      </div>
      <label for="adapter">LoRA adapter directory (optional)</label>
      <input id="adapter" placeholder="outputs/lora" />
      <label for="sentence">Source text</label>
      <textarea id="sentence">__SAMPLE__</textarea>
      <button class="primary" id="go">Translate</button>
      <pre id="translate-out">The first click downloads the Hugging Face checkpoint. CPU works for distilled models but is slow; a GPU is better.</pre>
    </section>
    <section id="langs">
      <table>
        <thead><tr><th>Tag</th><th>Language</th><th>Script</th></tr></thead>
        <tbody id="lang-body"></tbody>
      </table>
    </section>
  </main>
  <script>
    const LANGS = __LANGS__;
    const defaults = { src: "eng_Latn", tgt: "hin_Deva" };
    function fill(select, value) {
      select.innerHTML = LANGS.map(([code, name]) =>
        `<option value="${code}" ${code === value ? "selected" : ""}>${code} — ${name}</option>`
      ).join("");
    }
    ["src","tgt","p-src","p-tgt","t-src","t-tgt"].forEach((id) => {
      const prefer = id.includes("tgt") ? defaults.tgt : defaults.src;
      fill(document.getElementById(id), prefer);
    });
    document.getElementById("lang-body").innerHTML = LANGS.map(
      ([code, name, script]) => `<tr><td><code>${code}</code></td><td>${name}</td><td>${script}</td></tr>`
    ).join("");
    document.querySelectorAll("nav button").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        document.querySelectorAll("section").forEach((s) => s.classList.remove("visible"));
        document.getElementById(btn.dataset.tab).classList.add("visible");
      });
    });
    async function refreshRecipe() {
      const params = new URLSearchParams({
        src: document.getElementById("src").value,
        tgt: document.getElementById("tgt").value,
        size: document.getElementById("size").value,
        steps: document.getElementById("steps").value,
      });
      const res = await fetch("/api/recipe?" + params.toString());
      const data = await res.json();
      document.getElementById("recipe-out").textContent = data.command;
    }
    ["src","tgt","size","steps"].forEach((id) => document.getElementById(id).addEventListener("change", refreshRecipe));
    refreshRecipe();
    document.getElementById("convert").addEventListener("click", async () => {
      const train = document.getElementById("train-file").files[0];
      const dev = document.getElementById("dev-file").files[0];
      const out = document.getElementById("prepare-out");
      if (!train || !dev) { out.textContent = "Upload both a train and a dev file."; return; }
      const body = new FormData();
      body.append("train", train);
      body.append("dev", dev);
      body.append("src_lang", document.getElementById("p-src").value);
      body.append("tgt_lang", document.getElementById("p-tgt").value);
      const res = await fetch("/api/prepare", { method: "POST", body });
      out.textContent = JSON.stringify(await res.json(), null, 2);
    });
    document.getElementById("go").addEventListener("click", async () => {
      const out = document.getElementById("translate-out");
      out.textContent = "Translating…";
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: document.getElementById("sentence").value,
          src_lang: document.getElementById("t-src").value,
          tgt_lang: document.getElementById("t-tgt").value,
          adapter: document.getElementById("adapter").value,
        }),
      });
      const data = await res.json();
      out.textContent = data.translation || data.error || JSON.stringify(data, null, 2);
    });
  </script>
</body>
</html>
"""


def _page() -> str:
    langs = json.dumps(
        [[code, meta["name"], meta["script"]] for code, meta in LANGUAGES.items()],
        ensure_ascii=False,
    )
    return PAGE.replace("__LANGS__", langs).replace("__SAMPLE__", SAMPLE_EN)


def create_app() -> FastAPI:
    app = FastAPI(title="IndicTrans2 Fine-tune")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _page()

    @app.get("/api/recipe")
    def recipe(src: str, tgt: str, size: str = "distilled", steps: int = 400) -> dict:
        src = normalize_lang(src)
        tgt = normalize_lang(tgt)
        model = default_model(src, tgt, size)
        command = f"""# 1. Convert your parallel data
indictrans2-ft prepare \\
  --train data/sample/train.jsonl \\
  --dev data/sample/dev.jsonl \\
  --src-lang {src} --tgt-lang {tgt} \\
  --output-dir data/prepared

# 2. Fine-tune LoRA on {model}
indictrans2-ft train \\
  --data-dir data/prepared \\
  --src-lang {src} --tgt-lang {tgt} \\
  --model {model} \\
  --output-dir outputs/lora \\
  --max-steps {int(steps)} \\
  --batch-size 4 --grad-accum-steps 4

# 3. Translate with the adapter
indictrans2-ft translate \\
  --text "{SAMPLE_EN}" \\
  --src-lang {src} --tgt-lang {tgt} \\
  --adapter outputs/lora
"""
        return {"model": model, "command": command}

    @app.post("/api/prepare")
    async def prepare(
        train: UploadFile = File(...),
        dev: UploadFile = File(...),
        src_lang: str = Form(...),
        tgt_lang: str = Form(...),
    ):
        out = Path(tempfile.mkdtemp(prefix="indictrans2-"))
        train_path = out / (train.filename or "train.jsonl")
        dev_path = out / (dev.filename or "dev.jsonl")
        train_path.write_bytes(await train.read())
        dev_path.write_bytes(await dev.read())
        try:
            train_report = prepare_file(train_path, out, "train", src_lang, tgt_lang)
            dev_report = prepare_file(dev_path, out, "dev", src_lang, tgt_lang)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=400)
        src = normalize_lang(src_lang)
        tgt = normalize_lang(tgt_lang)
        return {
            "output_dir": str(out),
            "train": train_report.pairs["train"],
            "dev": dev_report.pairs["dev"],
            "skipped_train": train_report.skipped,
            "skipped_dev": dev_report.skipped,
            "base_model": default_model(src, tgt),
            "train_command": (
                f"indictrans2-ft train --data-dir {out} --src-lang {src} "
                f"--tgt-lang {tgt} --output-dir outputs/lora --max-steps 400 --batch-size 4"
            ),
        }

    @app.post("/api/translate")
    def translate(payload: dict) -> dict:
        text = (payload.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "Enter a sentence to translate."}, status_code=400)
        try:
            from indictrans2_finetune.infer import maybe_adapter_base, translate_batch

            src = normalize_lang(payload["src_lang"])
            tgt = normalize_lang(payload["tgt_lang"])
            adapter = (payload.get("adapter") or "").strip() or None
            model_name = default_model(src, tgt)
            if adapter:
                model_name = maybe_adapter_base(Path(adapter)) or model_name
            hyps = translate_batch([text], src, tgt, model_name=model_name, adapter=adapter)
            return {"translation": hyps[0] if hyps else ""}
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {
                    "error": (
                        f"{exc}. The first run downloads the Hugging Face checkpoint. "
                        "A GPU is recommended; CPU works for distilled 200M models but is slow."
                    )
                },
                status_code=500,
            )

    return app


def launch(host: str = "0.0.0.0", port: int = 43147) -> None:
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    launch()
