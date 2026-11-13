# FonadaLabs

Native Kapture V3 speech-to-text over `POST https://api.fonada.ai/v3/asr/transcribe`. Audio is processed in Fonada's cloud using customer-owned credits.

```python
import os
from esperanto import AIFactory

model = AIFactory.create_speech_to_text(
    "fonadalabs", "kapture-v3", {"api_key": os.environ["FONADALABS_API_KEY"]}
)
result = await model.atranscribe("lecture.mp3", language="hi")
print(result.text)
```

`FONADALABS_API_KEY` is also the environment fallback. `language` maps to multipart `language_id`; config `language` and then `FONADALABS_LANGUAGE` are fallbacks, defaulting to English (`en`). The endpoint does not promise automatic language detection. Both synchronous and asynchronous methods accept paths or binary streams. Prompt guidance is unsupported and raises ValueError.

Kapture V3 supports `en`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `or`, `pa`, `ta` and `te`. Send the code for the recording language. An unsupported code is rejected by Fonada before transcription.

Uploads must be nonempty and at most 25,000,000 bytes. Extract video audio and split larger recordings in the calling application's media layer. HTTP 429 waits 60 seconds and retries the same chunk once; a second failure stops. Other HTTP failures, including 502/503/504, and transport errors are not automatically retried: a timeout or gateway response may follow a successful billed request. Errors identify invalid keys, denied access, exhausted credits, upload limits, invalid audio/language and rate limits without echoing response bodies or secrets.

The response maps actual `text`, `transcript` or `transcription` fields, including the `result` envelope, to `TranscriptionResponse.text`. Timings, usage, duration and speaker labels remain absent. Model discovery returns `kapture-v3` statically and does not validate a key or consume credits.

An optional `base_url` accepts either `https://api.fonada.ai/v3/asr` or the complete `/v3/asr/transcribe` endpoint, with or without a trailing slash. The provider appends the transcription path only once.
