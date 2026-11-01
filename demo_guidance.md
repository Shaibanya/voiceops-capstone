# Two-minute demo guidance

No demo video or audio clip is supplied or generated. The brief asks for a demo
of no more than two minutes. Record your own voice, and do not record another
person without written consent. Add the final recording as `demo.mp4` only after
you have reviewed it; keep API keys and `.env` out of it.

Start the browser app with `python run.py web` and open
`http://127.0.0.1:5000`. Suggested screen and narration sequence:

1. **0:00–0:15 — Introduce the agent.** Show the browser interface and identify
   it as the ZentroMart AI assistant.
2. **0:15–0:45 — Grounded answer.** Ask one policy question, then a short
   follow-up. Show the reply and its cited section IDs.
3. **0:45–1:15 — Safe order lookup.** Provide a synthetic order ID, verify with
   the last four phone digits or delivery pincode, and show that the agent asks
   before details are exposed. Do not display the verification value in the
   recording.
4. **1:15–1:45 — Human handover or unknown.** Ask for a human colleague or ask
   something unsupported. Confirm a ticket only if the mock tool says it was
   created.
5. **1:45–2:00 — Voice.** Click **Record voice**, say a short question, then
   click **Stop & send**. Show its transcript and play the Gnani-generated
   spoken reply. The browser is push-to-talk, not a continuous live call.

Use only the supplied synthetic customer/order records. Test the live audio path
before recording the demo. If using a real Gnani or LLM key, keep the terminal
window and `.env` contents out of the recording.
