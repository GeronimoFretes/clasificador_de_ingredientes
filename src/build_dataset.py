# ──────────────────────────────────────────────────────────────────────────────
import io, asyncio
from pathlib import Path
from urllib.parse import quote_plus

import aiohttp, aiofiles
from PIL import Image
import torch, open_clip
from tqdm.asyncio import tqdm
from playwright.async_api import async_playwright

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG ─── you only tweak things in this box ────────────────────────────────
OUTPUT_DIR = (Path(__file__).parent / ".." / "data" / "raw").resolve()        # root folder for downloaded imgs
IMAGES_PER_CLASS   = 300                     # target after CLIP re‑ranking
SCROLL_ROUNDS      = 8                       # DuckDuckGo infinite‑scroll depth
SCROLL_PAUSE_MS    = 800
HEADLESS_BROWSER   = False                    # set False to watch the scraping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = {
    "manteca": [
        "manteca empaquetada",
        "manteca la serenisima 200g",
        "mantequilla barra",
    ],
    "banana": [
        "banana",
        "plátano amarillo sobre fondo blanco",
        "bananas en manojo",
    ],
    "harina": [
        "harina paquete",
        "harina bolsa 1 kg",
        "paquete de harina marca argentina",
    ],
    "leche": [
        "leche sachet 1 litro",
        "leche caja deslactosada",
        "leche entera en botella plastica",
        "leche larga vida argentina",
    ],
    "huevo_caja": [
        "maple de huevos 12",
        "huevos caja de cartón",
        "huevos pastoriles paquete",
    ],
    "huevo_suelto": [
        "huevo blanco mano",
        "un huevo marron",
        "un huevo blanco",
        "one egg in hand",
    ],
    "azucar": [
        "azúcar 1 kg paquete",
        "azucar bolsa 1 kilo",
        "azúcar refinado en paquete",
        "azúcar mascabo paquete",
    ],
}

CAPTION_LISTS = {
    "manteca": {
        "whitelist": ["manteca", "mantequilla"],
        "blacklist": [
            "medialuna",
            "croissant",
            "mantecado",
            "empanada",
            "figac",
            "pan",
            "crema",
            "dibuj",
            "cerdo",
            "carne",
            "kg",
            "cacao",
        ],
    },
    "banana": {
        "whitelist": ["banana", "plátano", "platano"],
        "blacklist": [
            "medialuna",
            "croissant",
            "mantecado",
            "empanada",
            "figac",
            "pan",
            "crema",
            "dibuj",
            "cerdo",
            "carne",
            "kg",
            "cacao",
            "dancing",
            "green",
            "hoja",
            "arbol",
            "árbol",
            "vector",
            "ilustración",
            "ilustracion",
            "tree",
        ],
    },
    "harina": {
        "whitelist": ["harina"],
        "blacklist": [
            "pan",
            "pizza",
            "pastel",
            "torta",
            "gallet",
            "cake",
            "bread",
            "figac",
            "harina integral",
            "maíz",
            "cornflour",
            "dibuj",
            "vector",
            "icono",
            "tortill",
        ],
    },
    "leche": {
        "whitelist": ["leche"],
        "blacklist": [
            "polvo",
            "powder",
            "condensada",
            "evaporada",
            "yogur",
            "yogurt",
            "queso",
            "cheese",
            "manteca",
            "butter",
            "café",
            "coffee",
            "shake",
            "batido",
            "smoothie",
            "helado",
            "ice cream",
            "vector",
            "icono",
            "dibuj",
            "vaca",
            "cow",
        ],
    },
    "huevo_caja": {
        "whitelist": ["huevo", "huevos"],
        "blacklist": [
            "frito",
            "fried",
            "revuelt",
            "omelette",
            "scrambled",
            "tortill",
            "huevo suelto",
            "single egg",
            "raw egg",
            "huevo cascara",
            "broken",
            "vector",
            "icono",
            "dibujo",
            "gallina",
            "pollo",
            "chicken",
        ],
    },
    "huevo_suelto": {
        "whitelist": ["huevo"],
        "blacklist": [
            "carton",
            "caja",
            "pack",
            "tray",
            "maple",
            "docena",
            "dozen",
            "6",
            "12",
            "frito",
            "fried",
            "revuelt",
            "scrambled",
            "omelette",
            "tortill",
            "vector",
            "icono",
            "dibujo",
            "art",
            "gallina",
            "pollo",
            "chicken",
            "eggs",
            "huevos",
        ],
    },
    "azucar": {
        "whitelist": ["azucar", "azúcar"],
        "blacklist": [
            "pastel",
            "cake",
            "gallet",
            "cookie",
            "postre",
            "dessert",
            "cuchara",
            "bowl",
            "spoon",
            "cucharada",
            "pile",
            "heap",
            "stevia",
            "edulcorante",
            "splenda",
            "honey",
            "miel",
            "vector",
            "icono",
            "dibuj",
            "art",
        ],
    },
}

CLIP_PROMPTS = {
    "manteca": ["manteca", "mantequilla"],
    "banana": ["banana", "plátano", "bananas", "plátanos"],
    "harina": ["harina"],
    "leche": ["carton de leche", "sachet de leche", "botella de leche"],
    "huevo_caja": ["carton de huevos", "maple de huevos", "huevos"],
    "huevo_suelto": ["huevo suelto", "single egg", "egg in hand"],
    "azucar": [
        "paquete de azúcar",
        "bag of sugar 1kg",
        "sugar package",
        "brown sugar bag",
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Chef‑IA Bot)",
    "Accept-Language": "es-AR,es;q=0.9",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

async def download_image_bytes(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=15) as r:
            if r.status == 200 and "image" in r.headers.get("content-type", ""):
                return await r.read()
    except Exception:
        return None

async def fetch_duckduckgo_candidates(query: str) -> list[dict]:
    """Return list[{url, caption}] for a single search query."""
    url = f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS_BROWSER)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")

        # Infinite‑scroll
        for _ in range(SCROLL_ROUNDS):
            await page.mouse.wheel(0, 1000)
            await page.wait_for_timeout(SCROLL_PAUSE_MS)

        figures = await page.query_selector_all("figure")
        results = []
        for fig in figures:
            img  = await fig.query_selector("img")
            if not img:
                continue
            src  = await img.get_attribute("src") or await img.get_attribute("data-src")
            if src and src.startswith("//"):
                src = "https:" + src
            if not src:
                continue
            cap_el  = await fig.query_selector("figcaption") or await fig.query_selector("span")
            caption = (await cap_el.inner_text()) if cap_el else ""
            results.append({"url": src, "caption": caption})
        await browser.close()
    return results

def caption_ok(cls: str, caption: str) -> bool:
    """Whitelist must appear; no blacklist token can appear."""
    caption_lc = caption.lower()
    wl_ok = any(tok in caption_lc for tok in CAPTION_LISTS[cls]["whitelist"])
    bl_ok = not any(tok in caption_lc for tok in CAPTION_LISTS[cls]["blacklist"])
    return wl_ok and bl_ok

def build_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.to(DEVICE).eval()
    return model, preprocess

# ──────────────────────────────────────────────────────────────────────────────
# CLIP re‑ranking
# ──────────────────────────────────────────────────────────────────────────────

async def clip_filter(cls: str, candidates: list[dict], prompts: list[str]) -> list[dict]:
    """
    Returns the top‑ranked metadata for a class after CLIP similarity.
    """
    model, preprocess = build_clip_model()
    # Download & preprocess images
    images, metas = [], []
    async with aiohttp.ClientSession(headers=HEADERS) as sess:
        for itm in tqdm(candidates, desc=f"⤓ img bytes ({cls})"):
            raw = await download_image_bytes(sess, itm["url"])
            if not raw:
                continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                images.append(preprocess(img))
                metas.append(itm)
            except Exception:
                continue

    if len(images) == 0:
        return []

    image_batch = torch.stack(images).to(DEVICE)          # (K, 3, 224, 224)

    # ---- correct tokenisation (77 tokens) ----
    txt_tokens = open_clip.tokenize(prompts).to(DEVICE)   # (P, 77)

    with torch.no_grad():
        img_emb = model.encode_image(image_batch).float()   # (K, D)
        txt_emb = model.encode_text(txt_tokens).float()     # (P, D)

    sims       = img_emb @ txt_emb.T                       # (K, P)
    best_sim   = sims.max(dim=1).values                    # (K,)

    k          = min(IMAGES_PER_CLASS, len(metas))
    keep_idx   = best_sim.topk(k).indices.cpu().tolist()
    return [metas[i] for i in keep_idx]

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

async def process_class(cls: str, queries: list[str]):
    # Fetch candidates from all queries
    all_candidates = []
    for q in queries:
        print(f"🔍 {cls}  – searching “{q}” …")
        cand = await fetch_duckduckgo_candidates(q)
        filtered = [c for c in cand if caption_ok(cls, c["caption"])]
        all_candidates.extend(filtered)

    print(f"• {cls}: {len(all_candidates)} candidates after caption filter")

    # CLIP ranking
    ranked = await clip_filter(cls, all_candidates, CLIP_PROMPTS[cls])
    print(f"• {cls}: {len(ranked)} kept after CLIP")

    # Download final images
    out_dir = OUTPUT_DIR / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(headers=HEADERS) as sess:
        tasks = []
        for i, meta in enumerate(ranked, 1):
            dest = out_dir / f"{cls}_{i:03d}.jpg"
            tasks.append(save_image(sess, meta["url"], dest))
        await asyncio.gather(*tasks)

async def save_image(session: aiohttp.ClientSession, url: str, dest: Path):
    raw = await download_image_bytes(session, url)
    if raw:
        try:
            async with aiofiles.open(dest, "wb") as f:
                await f.write(raw)
        except Exception:
            pass

async def main():
    for cls, queries in CLASSES.items():
        await process_class(cls, queries)

if __name__ == "__main__":
    asyncio.run(main())
