"""Deepgram Text-to-Speech provider implementation."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice

DEEPGRAM_VOICES: Dict[str, Voice] = {
    # Aura-2 English voices
    "aura-2-thalia-en": Voice(name="Thalia", id="aura-2-thalia-en", gender="FEMALE", language_code="en"),
    "aura-2-andromeda-en": Voice(name="Andromeda", id="aura-2-andromeda-en", gender="FEMALE", language_code="en"),
    "aura-2-helena-en": Voice(name="Helena", id="aura-2-helena-en", gender="FEMALE", language_code="en"),
    "aura-2-apollo-en": Voice(name="Apollo", id="aura-2-apollo-en", gender="MALE", language_code="en"),
    "aura-2-arcas-en": Voice(name="Arcas", id="aura-2-arcas-en", gender="MALE", language_code="en"),
    "aura-2-aries-en": Voice(name="Aries", id="aura-2-aries-en", gender="MALE", language_code="en"),
    "aura-2-amalthea-en": Voice(name="Amalthea", id="aura-2-amalthea-en", gender="FEMALE", language_code="en"),
    "aura-2-asteria-en": Voice(name="Asteria", id="aura-2-asteria-en", gender="FEMALE", language_code="en"),
    "aura-2-athena-en": Voice(name="Athena", id="aura-2-athena-en", gender="FEMALE", language_code="en"),
    "aura-2-atlas-en": Voice(name="Atlas", id="aura-2-atlas-en", gender="MALE", language_code="en"),
    "aura-2-aurora-en": Voice(name="Aurora", id="aura-2-aurora-en", gender="FEMALE", language_code="en"),
    "aura-2-callista-en": Voice(name="Callista", id="aura-2-callista-en", gender="FEMALE", language_code="en"),
    "aura-2-cora-en": Voice(name="Cora", id="aura-2-cora-en", gender="FEMALE", language_code="en"),
    "aura-2-cordelia-en": Voice(name="Cordelia", id="aura-2-cordelia-en", gender="FEMALE", language_code="en"),
    "aura-2-delia-en": Voice(name="Delia", id="aura-2-delia-en", gender="FEMALE", language_code="en"),
    "aura-2-draco-en": Voice(name="Draco", id="aura-2-draco-en", gender="MALE", language_code="en"),
    "aura-2-electra-en": Voice(name="Electra", id="aura-2-electra-en", gender="FEMALE", language_code="en"),
    "aura-2-harmonia-en": Voice(name="Harmonia", id="aura-2-harmonia-en", gender="FEMALE", language_code="en"),
    "aura-2-hera-en": Voice(name="Hera", id="aura-2-hera-en", gender="FEMALE", language_code="en"),
    "aura-2-hermes-en": Voice(name="Hermes", id="aura-2-hermes-en", gender="MALE", language_code="en"),
    "aura-2-hyperion-en": Voice(name="Hyperion", id="aura-2-hyperion-en", gender="MALE", language_code="en"),
    "aura-2-iris-en": Voice(name="Iris", id="aura-2-iris-en", gender="FEMALE", language_code="en"),
    "aura-2-janus-en": Voice(name="Janus", id="aura-2-janus-en", gender="MALE", language_code="en"),
    "aura-2-juno-en": Voice(name="Juno", id="aura-2-juno-en", gender="FEMALE", language_code="en"),
    "aura-2-jupiter-en": Voice(name="Jupiter", id="aura-2-jupiter-en", gender="MALE", language_code="en"),
    "aura-2-luna-en": Voice(name="Luna", id="aura-2-luna-en", gender="FEMALE", language_code="en"),
    "aura-2-mars-en": Voice(name="Mars", id="aura-2-mars-en", gender="MALE", language_code="en"),
    "aura-2-minerva-en": Voice(name="Minerva", id="aura-2-minerva-en", gender="FEMALE", language_code="en"),
    "aura-2-neptune-en": Voice(name="Neptune", id="aura-2-neptune-en", gender="MALE", language_code="en"),
    "aura-2-odysseus-en": Voice(name="Odysseus", id="aura-2-odysseus-en", gender="MALE", language_code="en"),
    "aura-2-ophelia-en": Voice(name="Ophelia", id="aura-2-ophelia-en", gender="FEMALE", language_code="en"),
    "aura-2-orion-en": Voice(name="Orion", id="aura-2-orion-en", gender="MALE", language_code="en"),
    "aura-2-orpheus-en": Voice(name="Orpheus", id="aura-2-orpheus-en", gender="MALE", language_code="en"),
    "aura-2-pandora-en": Voice(name="Pandora", id="aura-2-pandora-en", gender="FEMALE", language_code="en"),
    "aura-2-phoebe-en": Voice(name="Phoebe", id="aura-2-phoebe-en", gender="FEMALE", language_code="en"),
    "aura-2-pluto-en": Voice(name="Pluto", id="aura-2-pluto-en", gender="MALE", language_code="en"),
    "aura-2-saturn-en": Voice(name="Saturn", id="aura-2-saturn-en", gender="MALE", language_code="en"),
    "aura-2-selene-en": Voice(name="Selene", id="aura-2-selene-en", gender="FEMALE", language_code="en"),
    "aura-2-theia-en": Voice(name="Theia", id="aura-2-theia-en", gender="FEMALE", language_code="en"),
    "aura-2-vesta-en": Voice(name="Vesta", id="aura-2-vesta-en", gender="FEMALE", language_code="en"),
    "aura-2-zeus-en": Voice(name="Zeus", id="aura-2-zeus-en", gender="MALE", language_code="en"),
    # Aura-2 Spanish voices
    "aura-2-celeste-es": Voice(name="Celeste", id="aura-2-celeste-es", gender="FEMALE", language_code="es"),
    "aura-2-estrella-es": Voice(name="Estrella", id="aura-2-estrella-es", gender="FEMALE", language_code="es"),
    "aura-2-nestor-es": Voice(name="Nestor", id="aura-2-nestor-es", gender="MALE", language_code="es"),
    "aura-2-sirio-es": Voice(name="Sirio", id="aura-2-sirio-es", gender="MALE", language_code="es"),
    "aura-2-carina-es": Voice(name="Carina", id="aura-2-carina-es", gender="FEMALE", language_code="es"),
    "aura-2-alvaro-es": Voice(name="Alvaro", id="aura-2-alvaro-es", gender="MALE", language_code="es"),
    "aura-2-diana-es": Voice(name="Diana", id="aura-2-diana-es", gender="FEMALE", language_code="es"),
    "aura-2-aquila-es": Voice(name="Aquila", id="aura-2-aquila-es", gender="FEMALE", language_code="es"),
    "aura-2-selena-es": Voice(name="Selena", id="aura-2-selena-es", gender="FEMALE", language_code="es"),
    "aura-2-javier-es": Voice(name="Javier", id="aura-2-javier-es", gender="MALE", language_code="es"),
    "aura-2-agustina-es": Voice(name="Agustina", id="aura-2-agustina-es", gender="FEMALE", language_code="es"),
    "aura-2-antonia-es": Voice(name="Antonia", id="aura-2-antonia-es", gender="FEMALE", language_code="es"),
    "aura-2-gloria-es": Voice(name="Gloria", id="aura-2-gloria-es", gender="FEMALE", language_code="es"),
    "aura-2-luciano-es": Voice(name="Luciano", id="aura-2-luciano-es", gender="MALE", language_code="es"),
    "aura-2-olivia-es": Voice(name="Olivia", id="aura-2-olivia-es", gender="FEMALE", language_code="es"),
    "aura-2-silvia-es": Voice(name="Silvia", id="aura-2-silvia-es", gender="FEMALE", language_code="es"),
    "aura-2-valerio-es": Voice(name="Valerio", id="aura-2-valerio-es", gender="MALE", language_code="es"),
    # Aura-2 French voices
    "aura-2-agathe-fr": Voice(name="Agathe", id="aura-2-agathe-fr", gender="FEMALE", language_code="fr"),
    "aura-2-hector-fr": Voice(name="Hector", id="aura-2-hector-fr", gender="MALE", language_code="fr"),
    # Aura-2 German voices
    "aura-2-julius-de": Voice(name="Julius", id="aura-2-julius-de", gender="MALE", language_code="de"),
    "aura-2-viktoria-de": Voice(name="Viktoria", id="aura-2-viktoria-de", gender="FEMALE", language_code="de"),
    "aura-2-elara-de": Voice(name="Elara", id="aura-2-elara-de", gender="FEMALE", language_code="de"),
    "aura-2-aurelia-de": Voice(name="Aurelia", id="aura-2-aurelia-de", gender="FEMALE", language_code="de"),
    "aura-2-lara-de": Voice(name="Lara", id="aura-2-lara-de", gender="FEMALE", language_code="de"),
    "aura-2-fabian-de": Voice(name="Fabian", id="aura-2-fabian-de", gender="MALE", language_code="de"),
    "aura-2-kara-de": Voice(name="Kara", id="aura-2-kara-de", gender="FEMALE", language_code="de"),
    # Aura-2 Italian voices
    "aura-2-livia-it": Voice(name="Livia", id="aura-2-livia-it", gender="FEMALE", language_code="it"),
    "aura-2-dionisio-it": Voice(name="Dionisio", id="aura-2-dionisio-it", gender="MALE", language_code="it"),
    "aura-2-melia-it": Voice(name="Melia", id="aura-2-melia-it", gender="FEMALE", language_code="it"),
    "aura-2-elio-it": Voice(name="Elio", id="aura-2-elio-it", gender="MALE", language_code="it"),
    "aura-2-flavio-it": Voice(name="Flavio", id="aura-2-flavio-it", gender="MALE", language_code="it"),
    "aura-2-maia-it": Voice(name="Maia", id="aura-2-maia-it", gender="FEMALE", language_code="it"),
    "aura-2-cinzia-it": Voice(name="Cinzia", id="aura-2-cinzia-it", gender="FEMALE", language_code="it"),
    "aura-2-cesare-it": Voice(name="Cesare", id="aura-2-cesare-it", gender="MALE", language_code="it"),
    "aura-2-perseo-it": Voice(name="Perseo", id="aura-2-perseo-it", gender="MALE", language_code="it"),
    "aura-2-demetra-it": Voice(name="Demetra", id="aura-2-demetra-it", gender="FEMALE", language_code="it"),
    # Aura-2 Japanese voices
    "aura-2-fujin-ja": Voice(name="Fujin", id="aura-2-fujin-ja", gender="MALE", language_code="ja"),
    "aura-2-izanami-ja": Voice(name="Izanami", id="aura-2-izanami-ja", gender="FEMALE", language_code="ja"),
    "aura-2-uzume-ja": Voice(name="Uzume", id="aura-2-uzume-ja", gender="FEMALE", language_code="ja"),
    "aura-2-ebisu-ja": Voice(name="Ebisu", id="aura-2-ebisu-ja", gender="MALE", language_code="ja"),
    "aura-2-ama-ja": Voice(name="Ama", id="aura-2-ama-ja", gender="FEMALE", language_code="ja"),
    # Aura-2 Dutch voices
    "aura-2-rhea-nl": Voice(name="Rhea", id="aura-2-rhea-nl", gender="FEMALE", language_code="nl"),
    "aura-2-sander-nl": Voice(name="Sander", id="aura-2-sander-nl", gender="MALE", language_code="nl"),
    "aura-2-beatrix-nl": Voice(name="Beatrix", id="aura-2-beatrix-nl", gender="FEMALE", language_code="nl"),
    "aura-2-daphne-nl": Voice(name="Daphne", id="aura-2-daphne-nl", gender="FEMALE", language_code="nl"),
    "aura-2-cornelia-nl": Voice(name="Cornelia", id="aura-2-cornelia-nl", gender="FEMALE", language_code="nl"),
    "aura-2-hestia-nl": Voice(name="Hestia", id="aura-2-hestia-nl", gender="FEMALE", language_code="nl"),
    "aura-2-lars-nl": Voice(name="Lars", id="aura-2-lars-nl", gender="MALE", language_code="nl"),
    "aura-2-roman-nl": Voice(name="Roman", id="aura-2-roman-nl", gender="MALE", language_code="nl"),
    "aura-2-leda-nl": Voice(name="Leda", id="aura-2-leda-nl", gender="FEMALE", language_code="nl"),
    # Aura-1 English legacy voices
    "aura-asteria-en": Voice(name="Asteria (Aura-1)", id="aura-asteria-en", gender="FEMALE", language_code="en"),
    "aura-luna-en": Voice(name="Luna (Aura-1)", id="aura-luna-en", gender="FEMALE", language_code="en"),
    "aura-stella-en": Voice(name="Stella (Aura-1)", id="aura-stella-en", gender="FEMALE", language_code="en"),
    "aura-athena-en": Voice(name="Athena (Aura-1)", id="aura-athena-en", gender="FEMALE", language_code="en"),
    "aura-hera-en": Voice(name="Hera (Aura-1)", id="aura-hera-en", gender="FEMALE", language_code="en"),
    "aura-orion-en": Voice(name="Orion (Aura-1)", id="aura-orion-en", gender="MALE", language_code="en"),
    "aura-arcas-en": Voice(name="Arcas (Aura-1)", id="aura-arcas-en", gender="MALE", language_code="en"),
    "aura-perseus-en": Voice(name="Perseus (Aura-1)", id="aura-perseus-en", gender="MALE", language_code="en"),
    "aura-angus-en": Voice(name="Angus (Aura-1)", id="aura-angus-en", gender="MALE", language_code="en"),
    "aura-orpheus-en": Voice(name="Orpheus (Aura-1)", id="aura-orpheus-en", gender="MALE", language_code="en"),
    "aura-helios-en": Voice(name="Helios (Aura-1)", id="aura-helios-en", gender="MALE", language_code="en"),
    "aura-zeus-en": Voice(name="Zeus (Aura-1)", id="aura-zeus-en", gender="MALE", language_code="en"),
}


class DeepgramTextToSpeechModel(TextToSpeechModel):
    """Deepgram Text-to-Speech provider using the Aura model family."""

    PROVIDER = "deepgram"
    DEFAULT_MODEL = "aura-2-thalia-en"
    DEFAULT_ENCODING = "mp3"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "Deepgram API key not found. Set DEEPGRAM_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url or "https://api.deepgram.com",
            config=kwargs,
        )

        if self.base_url:
            self.base_url = self.base_url.rstrip("/")

        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "err_msg",
                    error_data.get("message", f"HTTP {response.status_code}"),
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Deepgram API error: {error_message}")

    def generate_speech(
        self,
        text: str,
        voice: str = DEFAULT_MODEL,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        encoding = kwargs.get("encoding", self.DEFAULT_ENCODING)

        params: Dict[str, Any] = {"model": voice, "encoding": encoding}
        for param in ("container", "sample_rate", "bit_rate", "speed"):
            if param in kwargs:
                params[param] = kwargs[param]

        response = self.client.post(
            f"{self.base_url}/v1/speak",
            headers=self._get_headers(),
            json={"text": text},
            params=params,
        )
        self._handle_error(response)

        audio_bytes = response.content

        audio_response = AudioResponse(
            audio_data=audio_bytes,
            content_type=f"audio/{encoding}",
            model=self.model_name,
            voice=voice,
            provider="deepgram",
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return audio_response

    async def agenerate_speech(
        self,
        text: str,
        voice: str = DEFAULT_MODEL,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        encoding = kwargs.get("encoding", self.DEFAULT_ENCODING)

        params: Dict[str, Any] = {"model": voice, "encoding": encoding}
        for param in ("container", "sample_rate", "bit_rate", "speed"):
            if param in kwargs:
                params[param] = kwargs[param]

        response = await self.async_client.post(
            f"{self.base_url}/v1/speak",
            headers=self._get_headers(),
            json={"text": text},
            params=params,
        )
        self._handle_error(response)

        audio_bytes = response.content

        audio_response = AudioResponse(
            audio_data=audio_bytes,
            content_type=f"audio/{encoding}",
            model=self.model_name,
            voice=voice,
            provider="deepgram",
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return audio_response

    @property
    def available_voices(self) -> Dict[str, Voice]:
        return dict(DEEPGRAM_VOICES)

    def _get_models(self) -> List[Model]:
        return []
