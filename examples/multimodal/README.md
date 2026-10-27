---
title: Multimodal Agents in 10xGraph - Image, Document, Audio and Video Input
description: Send images, PDFs, audio and video to a 10xGraph Python agent by URL, base64 or media-store reference, with OpenAI gpt-5-mini and Gemini, MultimodalConfig and MediaFetchPolicy.
section: Examples
group: "Multimodal and realtime"
order: 200
label: Multimodal input
updated: "2026-10-28"
faq:
  - question: Which provider should I use for which media type?
    answer: In these examples, gpt-5-mini handles images and PDFs, and gemini-2.5-flash handles audio and video as well as images. The converters send audio as inline base64 only, and video only reaches Gemini.
  - question: Why does a video URL get refused?
    answer: For Gemini the server downloads external URLs itself, under MediaFetchPolicy. The default policy allows images and PDFs only, so set allowed_content_types to include video types and, ideally, allowed_hosts.
  - question: When should I use a media store?
    answer: When you do not want large base64 payloads in messages and checkpoints. Store the bytes once, keep a graph://media/<key> reference in the message, and pass media_store to compile().
---

## What you will build

Three small scripts that send media to an agent. Each uses message content blocks (`ImageBlock`, `DocumentBlock`, `AudioBlock`, `VideoBlock`) that point at a `MediaRef`. A `MediaRef` can be an external URL, inline base64, or a reference into a media store.

The samples are generated in `sample_media.py` (a small PNG and a one second WAV tone), so the scripts need no asset files.

## Key concepts

- `MediaRef(kind="url" | "data" | "file_id")`: where the bytes live.
- `ImageBlock`, `DocumentBlock`, `AudioBlock`, `VideoBlock`: content blocks inside a `Message`, next to `TextBlock`.
- `MultimodalConfig`: per-agent limits such as `max_image_size_mb`, `max_image_dimension` and `ImageHandling`.
- `InMemoryMediaStore` and `LocalFileMediaStore`: store bytes and return a key. Reference it as `graph://media/<key>`.
- `compile(media_store=..., media_policy=...)`: wires the store in, and sets which external URLs the server may download.
- `MediaFetchPolicy`: allowed hosts, content types, size cap and timeout for downloads.

## Prerequisites

```bash
pip install "10xgraph[openai,google-genai]"
export OPENAI_API_KEY=...    # image_and_document.py
export GEMINI_API_KEY=...    # audio_and_video.py, media_store.py
```

## Which provider takes which media

| Media | gpt-5-mini (OpenAI) | gemini-2.5-flash (Google) |
|---|---|---|
| Image by URL | Yes, OpenAI fetches it | Yes, the server downloads it under `MediaFetchPolicy` |
| Image by base64 | Yes | Yes |
| PDF by URL | Yes, with `api_style="responses"` | Yes, downloaded under the policy |
| Audio | Not covered here | Yes, inline base64 |
| Video | No | Yes, inline base64, or URL under a policy that allows video |
| Media-store reference (`graph://media/...`) | Not supported for `gpt-5-mini` (see below) | Yes |

Known limits in the current library: the capability table behind media-store resolution has no entry for `gpt-5-mini`, so a `graph://media/...` reference sent to it raises `UnsupportedMediaInputError`. Use URL or base64 with OpenAI, and the media store with Gemini. Audio references into a media store are not resolved either, so send audio as base64.

## Run it

```bash
python examples/multimodal/image_and_document.py
python examples/multimodal/audio_and_video.py                 # generated tone
python examples/multimodal/audio_and_video.py talk.wav clip.mp4
python examples/multimodal/media_store.py
```

## How it works

An image by URL or base64 differs only in the `MediaRef`:

```python
MediaRef(kind="url", url=IMAGE_URL, mime_type="image/png")
MediaRef(kind="data", data_base64=b64(make_png()), mime_type="image/png")
```

A PDF is a `DocumentBlock`. Chat Completions has no PDF input, so that agent uses the Responses API:

```python
document_agent = ReactAgent(model="gpt-5-mini", api_style="responses", ...)
```

Audio and video go to Gemini as base64. The policy only matters for hosted files:

```python
policy = MediaFetchPolicy(allowed_hosts=("storage.googleapis.com",), allowed_content_types=("video/*",))
app = agent.compile(media_policy=policy)
```

With a media store, bytes are uploaded once and messages carry a short reference:

```python
key = await media_store.store(data=make_png(), mime_type="image/png")
ref = MediaRef(kind="url", url=f"{MEDIA_SCHEME}{key}", mime_type="image/png")
app = agent.compile(media_store=media_store, ...)
```

`media_store.py` runs the same message against `InMemoryMediaStore` and `LocalFileMediaStore`.

## Expected output

Illustrative output.

```text
[image-url] A transparent PNG test image: a figure on a checkerboard background.
[image-base64] A smooth gradient from red on the left to blue on the right.
[audio] A steady tone with no speech.
[memory] A gradient that shifts from red to blue.
```

## Using a different model

Change the model string and install the matching extra, for example `"claude-sonnet-5-5"` with `anthropic` for image and PDF input. Check what the new provider accepts before sending audio or video.

## Next steps

- [Realtime audio](../realtime_audio/)
- [RAG](../rag/)
- [Prebuilt ReAct agent](../quickstart/)
