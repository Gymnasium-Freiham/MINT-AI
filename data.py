# data.py
import json
import os
import re
import requests
from urllib.parse import quote

def load_training_data(file_path='./assets/training_data.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def append_data(training_data, new_data, key=None):
    # If caller provided a key, keep the original explicit behavior (unchanged).
    if key is not None:
        # ...existing explicit key-handling logic...
        if key == "comicSeries":
            for entry in new_data["comicSeries"]:
                question = f"Was ist die Comic-Serie '{entry['title']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "githubrepos":
            for entry in new_data["githubrepos"]:
                question = f"Welche URL hat '{entry['names']}'?"
                answer = entry['url']
                training_data.append({"question": question, "answer": answer})
        elif key == "dishes":
            for entry in new_data["dishes"]:
                question = f"Was ist das Gericht '{entry['name']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "books":
            for entry in new_data["books"]:
                question = f"Was ist das Buch '{entry['title']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "movies":
            for entry in new_data["movies"]:
                question = f"Was ist der Film '{entry['title']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "fruits":
            for entry in new_data["fruits"]:
                question = f"Was ist die Frucht '{entry['name']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "animals":
            for entry in new_data["animals"]:
                question = f"Was ist das Tier '{entry['name']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "windowsVersions":
            for entry in new_data["windowsVersions"]:
                question = f"Was ist Windows {entry['version']}?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "deutsch6klassebayern":
            for entry in new_data["deutsch6klassebayern"]:
                question = f"Was ist das Thema '{entry['topic']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "superMarioGames":
            for entry in new_data["superMarioGames"]:
                question = f"Was ist das Super Mario Spiel '{entry['title']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
        elif key == "informatik6klassebayern":
            for entry in new_data["informatik6klassebayern"]:
                question = f"Was ist das Thema '{entry['topic']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
                for example in entry.get('examples', []):
                    question = f"Was ist '{example['title']}'?"
                    answer = example['summary']
                    training_data.append({"question": question, "answer": answer})
        elif key == "mathematik6klassebayern":
            for entry in new_data["mathematik6klassebayern"]:
                question = f"Was ist das Thema '{entry['topic']}'?"
                answer = entry['description']
                training_data.append({"question": question, "answer": answer})
                for example in entry.get('examples', []):
                    question = f"Was ist '{example['title']}'?"
                    answer = example['summary']
                    training_data.append({"question": question, "answer": answer})
        else:
            # If the provided key isn't recognized, fallthrough to inference below
            pass

    # If key handling already added entries, return early
    # (This avoids double-processing when explicit branches handled the data)
    if len(training_data) > 0 and key is not None:
        return training_data

    # --- Inference mode: no key provided or unknown key ---
    # Heuristics: for each top-level list in new_data, inspect item dicts and pick sensible fields.
    def pick_field(item, candidates):
        for c in candidates:
            if c in item and isinstance(item[c], str) and item[c].strip():
                return c
        # fallback: first string field
        for k, v in item.items():
            if isinstance(v, str) and v.strip():
                return k
        return None

    # Candidate preference lists (deterministic)
    question_candidates = ['title', 'name', 'names', 'repo', 'repo_name', 'topic', 'question', 'question_text', 'id']
    answer_candidates = ['description', 'summary', 'url', 'answer', 'content', 'body', 'details']

    # If new_data is a list, treat as a single collection
    collections = {}
    if isinstance(new_data, list):
        collections['items'] = new_data
    elif isinstance(new_data, dict):
        # find list-valued entries; if none, try to wrap single dict as one item
        for k, v in new_data.items():
            if isinstance(v, list):
                collections[k] = v
        if not collections:
            # maybe new_data is a single list-like dict or single item
            collections['items'] = [new_data] if isinstance(new_data, dict) else []

    # Process each collection deterministically
    for coll_name, items in collections.items():
        for item in items:
            if not isinstance(item, dict):
                continue
            # repo/url special case
            if ('url' in item) and ('name' in item or 'names' in item or 'repo' in item):
                name_key = 'names' if 'names' in item else ('name' if 'name' in item else 'repo')
                question = f"Welche URL hat '{item.get(name_key)}'?"
                answer = item.get('url', '')
                training_data.append({"question": question, "answer": answer})
            else:
                # General case: infer question/answer from available fields
                question_field = pick_field(item, question_candidates)
                answer_field = pick_field(item, answer_candidates)
                if question_field and answer_field:
                    question = item[question_field]
                    answer = item[answer_field]
                    training_data.append({"question": question, "answer": answer})

    return training_data

def _extract_subject_from_question(question):
    # Avoid extracting a subject from math-like inputs or pure numeric tokens
    def _is_math_expression(s):
        s = (s or "").strip()
        if not s:
            return False
        # If contains digits and math operators or is composed only of digits/operators -> math
        if re.search(r'\d', s) and re.search(r'[\+\-\*\/\^=()]', s):
            return True
        if re.fullmatch(r'[\d\.\s\+\-\*\/\^\(\)]+', s):
            return True
        return False

    if _is_math_expression(question):
        return None

    # --- NEW: handle measurement-style questions like "Wie lang ist der Rüssel eines Elefanten?" ---
    m = re.search(r'wie\s+(lang|groß|hoch|schwer|alt)\s+(?:ist|sind)\s+(?:der|die|das|ein|eine)?\s*(.+?)\?*$', question, flags=re.IGNORECASE)
    if m:
        title = m.group(2).strip()
        # normalize common German constructs: "Rüssel eines Elefanten" -> keep as-is (Wikipedia often resolves)
        if title.isdigit() or len(title) < 2 or _is_math_expression(title):
            return None
        return title

    # Prefer quoted subject, then common German patterns "Was ist ... 'X'?" or "Was ist der/die/das X?"
    m = re.search(r'[\"\'‹›«»](.+?)[\"\'›‹«»]', question)
    if m:
        title = m.group(1).strip()
        # reject numeric/very short titles
        if title.isdigit() or len(title) < 3 or _is_math_expression(title):
            return None
        return title
    m = re.search(r'Was ist(?: der| die| das| ein| eine)?\s+["\']?(.+?)["\']?\??$', question, flags=re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        if title.isdigit() or len(title) < 3 or _is_math_expression(title):
            return None
        return title

    # fallback: pick last useful token but avoid numeric-only or very short tokens
    words = [w.strip() for w in re.findall(r"[A-Za-zÄÖÜäöüẞß0-9\-\s]+", question) if w.strip()]
    if words:
        candidate = words[-1].strip()
        if candidate.isdigit() or len(candidate) < 3 or _is_math_expression(candidate):
            return None
        return candidate
    return None


def _fetch_wikipedia_summary(title, prefer_langs=('de', 'en'), max_chars=1000):
    if not title:
        return None
    # respect environment flag to avoid network if set
    if os.environ.get("NO_CONNECTION") == "1":
        return None
    for lang in prefer_langs:
        try:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            resp = requests.get(url, timeout=6, headers={"User-Agent": "LATIN-AI-crawler/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                # summary field usually exists
                summary = data.get("extract") or data.get("description")
                if summary:
                    # truncate deterministically
                    return summary if len(summary) <= max_chars else summary[:max_chars].rsplit(' ', 1)[0] + "..."
        except Exception:
            # try next language
            continue
    return None


def augment_answers_with_wikipedia(training_data, prefer_langs=('de', 'en'), max_chars=1000):
    """
    For each QA entry, try to extract the subject from the question and fetch a Wikipedia summary.
    If found, append it to the answer text under a tag "[Wikipedia]".
    This is deterministic and skips network calls when NO_CONNECTION=1.
    """
    for entry in training_data:
        try:
            q = entry.get("question", "")
            subj = _extract_subject_from_question(q)
            if not subj:
                continue
            wiki = _fetch_wikipedia_summary(subj, prefer_langs=prefer_langs, max_chars=max_chars)
            if wiki:
                # Avoid duplicating if already included
                if "[Wikipedia]" not in entry.get("answer", ""):
                    entry["answer"] = (entry.get("answer", "") or "") + "\n\n[Wikipedia]: " + wiki
        except Exception:
            # keep training_data unchanged on errors
            continue
    return training_data


def load_and_append_data(training_data, file_path, key=None, crawl=False):
    """
    Load JSON and append. If crawl=True, run Wikipedia augmentation after append.
    """
    new_data = load_json_data(file_path)
    training_data = append_data(training_data, new_data, key)
    if crawl:
        training_data = augment_answers_with_wikipedia(training_data)
    return training_data

# Public wrappers for use at runtime
def fetch_wikipedia_summary(title, prefer_langs=('de', 'en'), max_chars=1000):
    """Return a Wikipedia summary for `title` or None. Respects NO_CONNECTION."""
    return _fetch_wikipedia_summary(title, prefer_langs=prefer_langs, max_chars=max_chars)

def fetch_wikipedia_variants(title, prefer_langs=('de','en'), max_chars=1000):
    """
    Try title and sensible variants (splits like "X eines Y", last token, english fallback).
    Returns first non-empty summary or None.
    """
    if not title:
        return None
    # try exact
    res = _fetch_wikipedia_summary(title, prefer_langs=prefer_langs, max_chars=max_chars)
    if res:
        return res
    # split constructs like "Rüssel eines Elefanten" -> try "Rüssel" and "Elefant"
    m = re.search(r'(.+?)\s+(?:von|des|der|die|das|eines|einer)\s+(.+)', title, flags=re.IGNORECASE)
    if m:
        parts = [m.group(1).strip(), m.group(2).strip()]
    else:
        # comma/parentheses or whitespace-separated fallback
        parts = re.split(r'[,\(\)]', title)
        parts = [p.strip() for p in parts if p.strip()]
    for p in parts:
        res = _fetch_wikipedia_summary(p, prefer_langs=prefer_langs, max_chars=max_chars)
        if res:
            return res
        last = p.split()[-1] if p.split() else p
        res = _fetch_wikipedia_summary(last, prefer_langs=prefer_langs, max_chars=max_chars)
        if res:
            return res
    # english-length fallback (limited scope)
    try:
        res = _fetch_wikipedia_summary(f"{title} length", prefer_langs=('en',), max_chars=max_chars)
        if res:
            return res
    except Exception:
        pass
    return None

def extract_subject_from_question(question):
    """Deterministically extract a subject from a question string (e.g. 'Was ist eine Giraffe?')."""
    return _extract_subject_from_question(question)

def fetch_wikipedia_page_text(title, prefer_langs=('de','en'), max_chars=3000):
    """
    Fetch the plaintext page extract via MediaWiki API (action=query&prop=extracts&explaintext).
    Returns a longer extract (up to max_chars) or None. Respects NO_CONNECTION.
    """
    if not title:
        return None
    if os.environ.get("NO_CONNECTION") == "1":
        return None
    for lang in prefer_langs:
        try:
            url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
                "format": "json",
                "titles": title
            }
            resp = requests.get(url, params=params, timeout=8, headers={"User-Agent": "LATIN-AI-crawler/1.0"})
            if resp.status_code != 200:
                continue
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                extract = page.get("extract")
                if extract:
                    # deterministic truncation
                    text = extract if len(extract) <= max_chars else extract[:max_chars].rsplit(' ', 1)[0] + "..."
                    return text
        except Exception:
            continue
    return None

def fetch_wiktionary_definition(title, prefer_langs=('de','en'), max_chars=1000):
    """
    Fetch a short definition/intro for `title` from Wiktionary (preferred langs).
    Returns plaintext intro (truncated to max_chars) or None. Respects NO_CONNECTION.
    """
    if not title:
        return None
    if os.environ.get("NO_CONNECTION") == "1":
        return None
    for lang in prefer_langs:
        try:
            url = f"https://{lang}.wiktionary.org/w/api.php"
            params = {
                "action": "query",
                "prop": "extracts",
                "exintro": "1",
                "explaintext": "1",
                "redirects": "1",
                "format": "json",
                "titles": title
            }
            resp = requests.get(url, params=params, timeout=8, headers={"User-Agent": "LATIN-AI-crawler/1.0"})
            if resp.status_code != 200:
                continue
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                extract = page.get("extract")
                if extract:
                    text = extract if len(extract) <= max_chars else extract[:max_chars].rsplit(' ', 1)[0] + "..."
                    return text
        except Exception:
            continue
    return None
