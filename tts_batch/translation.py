"""Small cached translation adapter used by language-aware backends.

Two engines, tried in this order:

1. Argos Translate (https://github.com/argosopentech/argos-translate) - a
   fully offline neural MT library (CTranslate2-based), pip-installable, no
   API key, no daily quota. Covers most pairs via automatic pivoting through
   English ("fr" -> "en" -> target); each language pair is a ~50-70MB model,
   downloaded once and cached permanently under ~/.local/share/argos-translate
   (i.e. free/unlimited after that one-time cost - the right fit for
   large-scale batches). Optional: skipped entirely if not installed.
2. MyMemory, via deep-translator - a real translation API, used only as a
   fallback for the pairs Argos has no package for (roughly a quarter of
   Piper's language families, e.g. Welsh, Georgian, Telugu at the time this
   was written). Its free daily quota (~5000 chars/day, ~50000 with a
   contact email) is what makes it unsuitable as the sole engine at scale -
   keeping its share of the workload small is the point of trying Argos
   first. (Not GoogleTranslator: its page-scraping approach is broken as of
   this writing - Google changed the markup of the page it scrapes, so every
   request raises TranslationNotFound regardless of language pair, verified
   directly against https://translate.google.com/m.)

MyMemory needs BCP-47-ish codes ("fr-FR", not "fr"), while the codes this
project passes around (Piper voice catalog language families, --language)
are bare ISO 639-1 codes ("fr"). _resolve_mymemory_code() bridges the two:
an exact match is used as-is, otherwise the first MyMemory-supported region
for that language family is picked (e.g. "fr" -> "fr-BE", the first
alphabetically - any region translates the language itself the same way).

At least one of argostranslate/deep-translator must be installed; both is
best (maximum coverage with minimum MyMemory quota usage).
"""

import json
import time
from pathlib import Path

# Piper's catalog uses generic "no" for Norwegian; Argos only ships a
# specific "nb" (Bokmal) package, close enough for this purpose - used only
# when looking up/installing Argos packages, not for cache keys or MyMemory.
ARGOS_CODE_ALIASES = {"no": "nb"}


class Translator:
    """Translate short texts and cache results across interrupted runs."""

    def __init__(self, source_language: str, cache_path: Path, contact_email: str | None = None):
        self.source_language = source_language
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.contact_email = contact_email

        self._argos_translate = self._init_argos()
        self._mymemory_translator_cls, self._mymemory_codes = self._init_mymemory()
        if self._argos_translate is None and self._mymemory_translator_cls is None:
            raise RuntimeError(
                "Translation requires 'argostranslate' (offline, free, no quota - "
                "recommended for large batches) and/or 'deep-translator' (used as a "
                "fallback via MyMemory). Install at least one: "
                "pip install argostranslate deep-translator"
            )

        self._argos_available_packages = None
        self._argos_ready_pairs: dict[tuple[str, str], tuple[str, str]] = {}
        self._mymemory_translators = {}

    @staticmethod
    def _init_argos():
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError:
            return None
        try:
            argostranslate.package.update_package_index()
        except Exception as error:
            print(f"Could not refresh the Argos Translate package index ({error!r}); "
                  "only already-installed language pairs will be available.")
        return argostranslate

    @staticmethod
    def _init_mymemory():
        try:
            from deep_translator import MyMemoryTranslator
            from deep_translator.constants import MY_MEMORY_LANGUAGES_TO_CODES
        except ImportError:
            return None, []
        return MyMemoryTranslator, sorted(set(MY_MEMORY_LANGUAGES_TO_CODES.values()))

    # --- Argos Translate -------------------------------------------------

    def _argos_find_package(self, from_code: str, to_code: str):
        if self._argos_available_packages is None:
            self._argos_available_packages = self._argos_translate.package.get_available_packages()
        return next(
            (
                p
                for p in self._argos_available_packages
                if p.from_code == from_code and p.to_code == to_code
            ),
            None,
        )

    def _argos_pair_installed(self, from_code: str, to_code: str) -> bool:
        installed = self._argos_translate.package.get_installed_packages()
        return any(p.from_code == from_code and p.to_code == to_code for p in installed)

    def _ensure_argos_ready(self, source: str, target: str) -> tuple[str, str] | None:
        """Make sure the package(s) needed for source->target (direct or
        pivoting through English) are installed, downloading any that
        aren't yet. Returns the (possibly alias-resolved) codes to actually
        pass to translate.translate(), or None if Argos has no path for this
        pair at all - the caller falls back to MyMemory in that case."""
        cached = self._argos_ready_pairs.get((source, target))
        if cached is not None:
            return cached

        resolved_source = ARGOS_CODE_ALIASES.get(source, source)
        resolved_target = ARGOS_CODE_ALIASES.get(target, target)

        if resolved_source == resolved_target:
            hops = []
        elif self._argos_find_package(resolved_source, resolved_target) is not None:
            hops = [(resolved_source, resolved_target)]
        elif resolved_source != "en" and resolved_target != "en":
            if self._argos_find_package(resolved_source, "en") and self._argos_find_package(
                "en", resolved_target
            ):
                hops = [(resolved_source, "en"), ("en", resolved_target)]
            else:
                return None
        else:
            return None

        installed_something_new = False
        for from_code, to_code in hops:
            if self._argos_pair_installed(from_code, to_code):
                continue
            package = self._argos_find_package(from_code, to_code)
            print(f"Downloading Argos Translate model for {from_code}->{to_code} (one-time)...")
            self._argos_translate.package.install_from_path(package.download())
            installed_something_new = True

        if installed_something_new:
            # argostranslate.translate.get_installed_languages() is
            # lru_cache'd AND its underlying PackageTranslation objects are
            # reused across calls via a module-level `installed_translates`
            # list - so a package installed mid-process leaves stale
            # Language object references, and pivot lookups through them
            # silently resolve to None (translate.translate() then crashes
            # with AttributeError: 'NoneType' object has no attribute
            # 'translate' - reproduced and confirmed while building this).
            # Clearing both forces a fully fresh rebuild of the translation
            # graph from what's actually installed on disk.
            self._argos_translate.translate.get_installed_languages.cache_clear()
            self._argos_translate.translate.installed_translates.clear()

        resolved = (resolved_source, resolved_target)
        self._argos_ready_pairs[(source, target)] = resolved
        return resolved

    def _translate_argos(self, text: str, target_language: str) -> str | None:
        if self._argos_translate is None:
            return None

        try:
            resolved = self._ensure_argos_ready(self.source_language, target_language)
        except Exception as error:
            # A genuine problem (e.g. a flaky package download), as opposed to
            # Argos simply having no package for this pair - worth surfacing
            # instead of silently falling back to MyMemory with no trace.
            print(f"Argos Translate setup failed for {self.source_language!r}->"
                  f"{target_language!r} ({error!r}), falling back to MyMemory.")
            return None
        if resolved is None:
            return None  # no Argos path for this pair - the normal, silent case

        try:
            translated = self._argos_translate.translate.translate(text, *resolved)
            return translated or None
        except Exception as error:
            print(f"Argos Translate failed for {self.source_language!r}->{target_language!r} "
                  f"({error!r}), falling back to MyMemory.")
            return None

    # --- MyMemory ----------------------------------------------------------

    def _resolve_mymemory_code(self, language: str) -> str:
        """Map a bare/underscore code ('fr', 'fr_FR') to a MyMemory-supported
        BCP-47-ish code ('fr-FR'), preferring an exact match."""
        normalized = language.replace("_", "-")
        if normalized in self._mymemory_codes:
            return normalized
        prefix = normalized.split("-")[0].lower()
        for code in self._mymemory_codes:
            if code.split("-")[0].lower() == prefix:
                return code
        raise RuntimeError(
            f"MyMemory has no supported language code for {language!r}. Supported "
            f"prefixes: {sorted({c.split('-')[0] for c in self._mymemory_codes})}"
        )

    def _get_mymemory_translator(self, target_language: str):
        translator = self._mymemory_translators.get(target_language)
        if translator is None:
            kwargs = {"email": self.contact_email} if self.contact_email else {}
            translator = self._mymemory_translator_cls(
                source=self._resolve_mymemory_code(self.source_language),
                target=self._resolve_mymemory_code(target_language),
                **kwargs,
            )
            self._mymemory_translators[target_language] = translator
        return translator

    def _translate_mymemory(self, text: str, target_language: str) -> str:
        last_error = None
        for attempt in range(3):
            try:
                translated = self._get_mymemory_translator(target_language).translate(text)
                if not translated:
                    raise RuntimeError("translation returned no text")
                return translated
            except Exception as error:
                last_error = error
                self._mymemory_translators.pop(target_language, None)
                if attempt < 2:
                    time.sleep(attempt + 1)
        raise RuntimeError(
            f"Could not translate text to {target_language!r} after 3 attempts: "
            f"{last_error!r}; text={text!r}"
        ) from last_error

    # --- cache + public API -------------------------------------------------

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot read translation cache {self.cache_path}: {error}") from error
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(text, str) for key, text in value.items()
        ):
            raise RuntimeError(f"Translation cache {self.cache_path} must contain a JSON object")
        return value

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def translate(self, text: str, target_language: str) -> str:
        if target_language == self.source_language:
            return text

        cache_key = f"{self.source_language}\t{target_language}\t{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        translated = self._translate_argos(text, target_language)
        if translated is None:
            if self._mymemory_translator_cls is None:
                raise RuntimeError(
                    f"Argos Translate has no {self.source_language!r}->{target_language!r} "
                    "path and no MyMemory fallback is installed (pip install deep-translator)."
                )
            translated = self._translate_mymemory(text, target_language)

        self.cache[cache_key] = translated
        self._save_cache()
        return translated
