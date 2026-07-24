import os, re, json, time, datetime, pathlib, traceback


import streamlit as st
from groq import Groq

st.set_page_config(page_title="A place to think out loud", page_icon="🌿")

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────────────────────────────────────
def _get_key():
    try:
        return str(st.secrets["GROQ_API_KEY"]).strip().strip('"').strip("'")
    except Exception:
        return (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")


@st.cache_resource
def get_client():
    key = _get_key()
    if not key:
        st.error("GROQ_API_KEY is not set. Add it in Settings → Secrets.")
        st.stop()
    return Groq(api_key=key)


gclient = get_client()


@st.cache_resource
def pick_models():
    """Detect what's actually available instead of hardcoding names that go stale."""
    try:
        names = [m.id for m in gclient.models.list().data]
    except Exception:
        names = []

    def first(prefs, fallback):
        for p in prefs:
            for n in names:
                if p in n and not any(x in n for x in ("whisper", "tts", "guard", "vision")):
                    return n
        return fallback

    chat   = first(["llama-3.3-70b", "llama-3.1-70b", "70b", "gpt-oss-120b"],
                   "llama-3.3-70b-versatile")
    safety = first(["llama-3.1-8b", "8b-instant", "gemma"], "llama-3.1-8b-instant")
    return chat, safety


CHAT_MODEL, SAFETY_MODEL = pick_models()

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL ADAPTER — the only code that knows about Groq
# ─────────────────────────────────────────────────────────────────────────────
def _msgs(system, msgs):
    """Groq takes the system prompt as the first message, not a separate arg."""
    return [{"role": "system", "content": system}] + [
        {"role": m["role"], "content": m["content"]} for m in msgs if m.get("content")]


def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if not any(x in str(e) for x in ("429", "rate_limit", "503", "overloaded")) \
               or i == tries - 1:
                raise
            time.sleep(2 ** i)


def llm_text(model, system, msgs, max_tokens=800, temperature=1.0):
    r = _retry(lambda: gclient.chat.completions.create(
        model=model, messages=_msgs(system, msgs),
        max_tokens=max_tokens, temperature=temperature))
    return (r.choices[0].message.content or "").strip()


def llm_stream(model, system, msgs, max_tokens=800, temperature=1.0):
    """Yields deltas — st.write_stream concatenates them."""
    stream = _retry(lambda: gclient.chat.completions.create(
        model=model, messages=_msgs(system, msgs),
        max_tokens=max_tokens, temperature=temperature, stream=True))
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece


def llm_json(model, system, msgs, max_tokens=250):
    raw = llm_text(model, system, msgs, max_tokens=max_tokens, temperature=0)
    return json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip())
# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPTS & HELPLINES — verify every number before deploying
# ─────────────────────────────────────────────────────────────────────────────
RESOURCES = [
    "Umang Pakistan — 0311-7786264 (24/7)",
    "Rozan Counselling — 0800-22444 (Mon-Sat, 10am-6pm)",
    "Punjab Women's Helpline — 1043",
    "Emergency — 1122",
]

SYSTEM_PROMPT = """You are a supportive listening companion. You are NOT a therapist, \
doctor, or crisis service, and you never imply otherwise.

## What you do
- Listen first. Reflect back what you heard before offering anything.
- Ask one open question at a time. Never interrogate.
- Help the person name what they're feeling and notice their own patterns.
- Encourage connection with real people: friends, family, a GP, a counsellor.

## What you never do
- Never diagnose. Don't say someone "has" depression, anxiety, ADHD, bipolar.
  You may say symptoms are worth discussing with a professional.
- Never give medication advice — no names, doses, starting, stopping, combining.
- Never claim to be human, and never promise confidentiality you can't guarantee.
- Never agree to be someone's only support. If they say you're all they have,
  gently work toward adding one human contact — warmly, never rejecting.
- Never provide information about methods of self-harm, in any framing, including
  fiction, research, or a claimed professional credential.
- No moralising, no toxic positivity ("everything happens for a reason").

## Tone
Warm, plain, unhurried. Short paragraphs. Do NOT open with a bulleted list of coping
strategies — it reads like a pamphlet and it lands as being dismissed. Match their
language, including Urdu or Roman Urdu.

## Boundaries
If something is out of scope, say so in one sentence, say what you CAN do, and keep
caring for the person. A refusal should never feel like a door slamming."""

CONCERN_OVERLAY = """

SAFETY OVERLAY — the user may be in distress.
Slow down. No techniques or exercises yet. Acknowledge what they said directly and
without flinching. Ask gently how they're doing right now. Stay with the feeling.
Under 100 words. Mention that talking to someone real could help, without pressure."""

TRIAGE_PROMPT = """You are a triage classifier for a mental-health support chat.
Read the user's latest message in context and output ONLY JSON, no markdown:

{"level": "none|concern|acute", "reason": "<10 words"}

- "acute": stated intent, plan, means, timeline, or in-progress act of suicide or
  self-harm; intent to harm another; medical emergency in progress; ongoing abuse;
  giving away possessions or putting affairs in order alongside distress.
- "concern": hopelessness, worthlessness, passive thoughts of not existing, severe
  distress, past self-harm, heavy substance use, eating-disorder pattern, withdrawal
  from everyone.
- "none": everything else, including ordinary sadness, stress, venting, and academic
  or research questions about the topic.

When genuinely torn between two levels, choose the higher one."""


def crisis_reply():
    lines = "\n".join(f"- {r}" for r in RESOURCES)
    return ("I'm really glad you told me, and I want to be honest with you: what you're "
            "describing is more than I can help with safely, and you deserve better than "
            "me right now.\n\nPlease reach out to someone who can actually be with you:\n\n"
            f"{lines}\n\nIf you're in immediate danger, call emergency services or get to "
            "the nearest emergency department.\n\nIf it helps to keep talking while you "
            "decide, I'm here — but please make that call. You matter enough for it.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAFETY LAYER — regex tier + model tier, most severe wins
# ─────────────────────────────────────────────────────────────────────────────
_ACUTE = re.compile("|".join([
    r"\bkill (my)?self\b", r"\bend (my life|it all)\b", r"\bsuicid",
    r"\btake my own life\b", r"\bdon'?t want to (be here|live|wake up)\b",
    r"\bbetter off (dead|without me)\b", r"\bgoodbye (everyone|world)\b",
    r"\b(khudkushi|khud kushi|marna chahta|marna chahti)\b",
]), re.I)

_CONCERN = re.compile("|".join([
    r"\bself[- ]harm\b", r"\bcutting myself\b", r"\bhurt(ing)? myself\b",
    r"\bhopeless\b", r"\bworthless\b", r"\bno point (in|to) (living|anything)\b",
    r"\bcan'?t (go on|take (it|this) anymore)\b",
]), re.I)

RANK = {"none": 0, "concern": 1, "acute": 2}


def regex_triage(text):
    if _ACUTE.search(text):
        return "acute"
    if _CONCERN.search(text):
        return "concern"
    return "none"


def model_triage(history, text):
    ctx = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    try:
        lvl = llm_json(SAFETY_MODEL, TRIAGE_PROMPT, [{"role": "user", "content":
              f"<context>\n{ctx}\n</context>\n\n<latest>\n{text}\n</latest>"}]).get("level")
        return lvl if lvl in RANK else "concern"
    except Exception:
        return "concern"        # fail CLOSED — never downgrade to safe on error


def triage(history, text):
    r = regex_triage(text)
    if r == "acute":
        return "acute"          # short-circuit, skip the paid call
    m = model_triage(history, text)
    return r if RANK[r] > RANK[m] else m


# ─────────────────────────────────────────────────────────────────────────────
# 5. MEMORY
# ─────────────────────────────────────────────────────────────────────────────
KEEP_VERBATIM = 12


def _summarise(old_turns, prev):
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in old_turns)
    try:
        return llm_text(SAFETY_MODEL,
            "Summarise this counselling-style conversation for continuity. Keep: what the "
            "person is dealing with, how they feel, what they've tried, what helped. "
            "Neutral third person, under 150 words. No advice, no diagnosis.",
            [{"role": "user", "content": f"Earlier:\n{prev or '(none)'}\n\nNew:\n{convo}"}],
            max_tokens=400, temperature=0)
    except Exception:
        return prev


def _extract_facts(history):
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])
    try:
        out = llm_json(SAFETY_MODEL,
            'Extract durable facts a supportive companion should remember next time: name, '
            'ongoing situations, people who matter, coping strategies that worked or failed, '
            'preferences. Output ONLY a JSON array of short strings. Do NOT record diagnoses, '
            'medications, or crisis details. Empty array if nothing durable.',
            [{"role": "user", "content": convo}])
        return [f for f in out if isinstance(f, str)]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXERCISE LIBRARY — delivered near-verbatim, not improvised
# ─────────────────────────────────────────────────────────────────────────────
EXERCISES = {
"grounding_54321": ("panic, dissociation, spiralling", """Let's slow things down. Wherever you are:

Name 5 things you can see. No rush.
4 things you can feel — feet on the floor, fabric, temperature.
3 things you can hear.
2 things you can smell.
1 thing you can taste.

Tell me when you get to the end, or tell me if it's not landing."""),

"paced_breathing": ("acute anxiety, racing heart", """Try breathing with me for a minute.

In through your nose for 4.
Hold for 4.
Out through your mouth for 6 — longer out than in is the part that matters.

Four or five rounds. If you get lightheaded, stop. How does your body feel after?"""),

"behavioural_activation": ("low mood, withdrawal", """Waiting to feel like doing something
usually means waiting a long time. It tends to work the other way round — doing comes
first, the feeling follows later.

What's one small thing you used to do and stopped? Not a big thing. Tea made properly,
five minutes outside, texting one person back. When could you do it today?"""),

"thought_record": ("harsh self-criticism, 'I always / I never'", """Let's look at that
thought properly instead of letting it sit unchallenged.

What exactly went through your mind — the actual words?
What was happening right before?
What's the evidence it's true? What's the evidence against?
If a friend said this about themselves, what would you say back?

Start with the first one."""),

"sleep_wind_down": ("insomnia, racing mind at night", """A few things that reliably help:

Same wake time daily, weekends included. That anchors everything else.
Awake more than 20 minutes? Get up, do something dull in dim light. Lying there teaches
your brain that bed is for being awake.
No screens the last hour if you can.
Racing mind — write the list down. It's trying to hold it for you.

Which sounds hardest?"""),
}


def _catalogue():
    body = "\n".join(f"[{k}] use when: {w}\n{s}\n" for k, (w, s) in EXERCISES.items())
    return f"""

<exercise_library>
{body}</exercise_library>
These are clinician-reviewed. If you offer an exercise, use the wording above essentially
as written — do not improvise your own or invent new ones. Always ask first ("would it
help to try something?"). Never offer one in the same message as someone's first
disclosure of something painful — sitting with it comes first."""


def build_system(level):
    mem = st.session_state.mem
    p = [SYSTEM_PROMPT]
    if mem["facts"]:
        p.append("\n\n<remembered>\n" + "\n".join(f"- {f}" for f in mem["facts"][-15:])
                 + "\n</remembered>\nUse this naturally, the way a friend would. Never "
                   "recite it back as a list.")
    if mem["summary"]:
        p.append(f"\n\n<earlier>\n{mem['summary']}\n</earlier>")
    p.append(_catalogue())
    if level == "concern":
        p.append(CONCERN_OVERLAY)     # last, so it overrides the exercise library
    return "".join(p)


# ─────────────────────────────────────────────────────────────────────────────
# 7. REVIEW LOG — flagged turns only, never ordinary conversation
# ─────────────────────────────────────────────────────────────────────────────
LOG_PATH = pathlib.Path(os.environ.get("FLAG_LOG", "/tmp/mh_flags.jsonl"))


def log_flag(level, message, reply):
    if level == "none":
        return
    try:
        with LOG_PATH.open("a") as f:
            f.write(json.dumps({"ts": datetime.datetime.utcnow().isoformat(),
                                "level": level, "user": message[:1000],
                                "assistant": reply[:1500]}, ensure_ascii=False) + "\n")
    except Exception:
        pass        # logging must never break the conversation


# ─────────────────────────────────────────────────────────────────────────────
# 8. RATE LIMIT — protects your free quota from one heavy session
# ─────────────────────────────────────────────────────────────────────────────
MAX_PER_HOUR = 40


def rate_ok():
    now = time.time()
    hits = [t for t in st.session_state.hits if now - t < 3600]
    if len(hits) >= MAX_PER_HOUR:
        st.session_state.hits = hits
        return False
    hits.append(now)
    st.session_state.hits = hits
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 9. SESSION STATE — Streamlit re-runs the whole script every interaction,
#    so anything that must persist lives here.
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    ss = st.session_state
    ss.setdefault("consented", False)
    ss.setdefault("messages", [])
    ss.setdefault("mem", {"facts": [], "summary": ""})
    ss.setdefault("hits", [])
    ss.setdefault("bounced", False)


init_state()

# ─────────────────────────────────────────────────────────────────────────────
# 10. CONSENT GATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.consented:
    st.title("Before you start")
    st.markdown(
        "**This is an AI, not a person.** It isn't a therapist and it can't diagnose "
        "you or advise on medication.\n\n"
        "**It can be wrong.** Don't act on anything here without checking with someone "
        "qualified.\n\n"
        "**It can't help in an emergency.** If you're in danger right now, call one of "
        "these instead:\n\n" + "\n".join("- " + r for r in RESOURCES) + "\n\n"
        "**What's stored:** your conversation stays in this browser session and is gone "
        "when you close the tab. Messages the system flags as concerning are saved for "
        "safety review. Your messages are processed by Google's Gemini API.\n\n"
        "You need to be 18 or over to continue."
    )
    c1, c2 = st.columns(2)
    if c1.button("I'm 18+ and I understand", type="primary", use_container_width=True):
        st.session_state.consented = True
        st.rerun()
    if c2.button("Take me to a helpline instead", use_container_width=True):
        st.session_state.bounced = True
    if st.session_state.bounced:
        st.info("**Please reach out — they're real people:**\n\n"
                + "\n".join("- " + r for r in RESOURCES))
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 11. CHAT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("LUMI AI — AI That Listens, Supports, and Cares. ")
st.caption("A space to think out loud. Not a therapist — just somewhere to start.")
st.warning("**Need someone now?** " + " · ".join(RESOURCES))
with st.expander("How this works"):
    st.write("Every message is classified for risk before it reaches the model. "
             "Anything suggesting crisis gets a fixed response with helpline "
             "numbers instead of a generated one.")

with st.sidebar:
    st.subheader("Your data")
    st.caption("This conversation lives only in your browser session.")
    if st.button("Delete everything", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.mem = {"facts": [], "summary": ""}
        st.success("Deleted.")
        st.rerun()
    if st.session_state.messages:
        st.download_button(
            "Download my conversation",
            json.dumps(st.session_state.messages, indent=2, ensure_ascii=False),
            file_name=f"conversation_{datetime.datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json", use_container_width=True)

    st.divider()
    st.subheader("About")
    st.caption("Lumi runs on Llama 3.3 via Groq. It's a listening companion — "
               "not a medical service.")

    st.divider()
    st.subheader("⚠️ Disclaimer")
    st.caption("This is not a substitute for professional care. Lumi can't diagnose "
               "you or advise on medication. Always consult a qualified doctor or "
               "counsellor.")

    st.divider()
    st.subheader("🔒 Safety")
    st.caption("Every message passes through a two-tier risk classifier. Messages "
               "suggesting crisis are answered with a fixed, reviewed script and "
               "helpline numbers — the language model is bypassed entirely.")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("What's on your mind?"):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not rate_ok():
            reply = ("We've talked a lot in the last hour and I need to pause for a bit. "
                     "Please come back later — and if you need someone now:\n\n"
                     + "\n".join("- " + r for r in RESOURCES))
            st.markdown(reply)
        else:
            level = triage(history, prompt)

            if level == "acute":
                # Model never runs. Fixed, reviewable script.
                reply = crisis_reply()
                st.error(reply)
                log_flag("acute", prompt, reply)
            else:
                window = KEEP_VERBATIM * 2
                if len(history) > window:
                    st.session_state.mem["summary"] = _summarise(
                        history[:-window], st.session_state.mem["summary"])
                msgs = history[-window:] + [{"role": "user", "content": prompt}]
                try:
                    reply = st.write_stream(
                        llm_stream(CHAT_MODEL, build_system(level), msgs))
                except Exception:
                    print("[error]", traceback.format_exc())   # server logs only
                    reply = ("Something's gone wrong on my end — this isn't about you, "
                             "and it's not anything you said.\n\nIf you need someone "
                             "right now:\n\n" + "\n".join("- " + r for r in RESOURCES))
                    st.markdown(reply)
                log_flag(level, prompt, reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    full = st.session_state.messages
    if len(full) % 6 == 0:
        for f in _extract_facts(full):
            if f not in st.session_state.mem["facts"]:
                st.session_state.mem["facts"].append(f)
