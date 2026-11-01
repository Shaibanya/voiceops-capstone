# Sample outputs

Run `python run.py evaluate --output .\sample_outputs` to generate the offline
practice evaluation. This measures local retrieval/decision baselines, not
generated-answer quality.

After configuring a Groq-compatible LLM key and Gnani key, run
`python run.py evaluate --with-llm --output .\sample_outputs` for the live
prompt-only versus RAG comparison. Provider rate limits or request errors can
stop the run before the output is refreshed.

Use `python run.py web` to open the browser interface. It supports browser
microphone push-to-talk and audio-file upload, as well as folder-based audio
processing via `python run.py batch --input .\my_audio --output .\outputs`. No
audio files are included; use only your own recording or audio you have
permission to process. This folder contains no fabricated audio or
generated-answer scores.
