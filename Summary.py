"""
Abstractive Text Summarization  -  modern window (Python only)
Same three modules as the project report:
  1. Generate & Load Transformer Model
  2. Preprocess Text
  3. Generate Abstractive Summary

Why the window opens fast: heavy libraries (torch, transformers, nltk) are loaded
in the background AFTER the window is shown.
"""
import re
import time
import threading
import io
import os
import queue
import random
from datetime import datetime
from string import punctuation
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# t5-small  : fastest, weakest (often copies the input)
# t5-base   : better rewriting, ~3x slower on CPU              <- default
# facebook/bart-large-cnn : best quality, slowest, ~1.6 GB download
GEMINI_CHOICE = 'Gemini (online)'      # needs internet + API key; the text is sent to Google
GEMINI_MODEL = 'gemini-3.5-flash'     # change here if Google renames the model
MODEL_CHOICES = ['t5-base', 't5-small', 'facebook/bart-large-cnn', GEMINI_CHOICE]
# Speed = number of beams. More beams = better text but slower (each beam is extra work).
SPEED_CHOICES = {'Fast (1 beam)': 1, 'Balanced (2 beams)': 2, 'Best quality (4 beams)': 4}
CHUNK_TOKENS = 400          # long text is split into pieces of about this size
MODEL_LIMIT = 512           # maximum tokens the model can read at once

# Filled in by the background start-up thread
torch = nltk = AutoTokenizer = AutoModelForSeq2SeqLM = None
StopFlag = None
WordCloud = None             # optional: pip install wordcloud
gemini_key = ''
stop_words = lemmatizer = None
device = None
model = tokenizer = model_name = None
libs_ready = False
cancel_event = threading.Event()
step_callback = None          # called by the decoder at every step (for the Thinking panel)
ui_queue = queue.Queue()      # worker threads put functions here; the window runs them


FALLBACK_STOP_WORDS = ('a an the and or but if of at by for with about to from in out on off over under again then once '
                       'here there when where why how all any both each few more most other some such no nor not only '
                       'own same so than too very can will just is are was were be been being have has had do does did '
                       'i me my we our you your he him his she her it its they them their this that these those as '
                       'what which who whom while into through during before after above below up down').split()


class Cancelled(Exception):
    pass


def init_libs():
    """Runs in a background thread: imports heavy libraries only once."""
    global torch, nltk, AutoTokenizer, AutoModelForSeq2SeqLM, StopFlag
    global stop_words, lemmatizer, device, libs_ready, WordCloud
    # NLTK is optional: if its data is broken, the app still works with built-in fallbacks.
    try:
        import nltk as _nltk
        nltk = _nltk
        for pkg, path in (('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet'),
                          ('punkt', 'tokenizers/punkt'), ('punkt_tab', 'tokenizers/punkt_tab')):
            try:
                nltk.data.find(path)                # already on disk: no network needed
            except Exception:
                try:
                    nltk.download(pkg, quiet=True)  # only the first time
                except Exception:
                    pass
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize('tests')               # fails now (not later) if wordnet is broken
    except Exception:
        stop_words = set(FALLBACK_STOP_WORDS)
        lemmatizer = None

    import torch as _torch
    import transformers
    from transformers import AutoTokenizer as _T, AutoModelForSeq2SeqLM as _M, StoppingCriteria
    torch, AutoTokenizer, AutoModelForSeq2SeqLM = _torch, _T, _M
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    new_api = v >= (4, 39)      # newer versions expect a tensor, older ones a bool

    class _StopFlag(StoppingCriteria):
        """Called at every decoder step: lets Stop interrupt, and feeds the Thinking panel."""
        def __call__(self, input_ids, scores, **kwargs):
            flag = cancel_event.is_set()
            cb = step_callback
            if cb is not None:
                step = int(input_ids.shape[-1])
                draft = ''
                if step % 3 == 0:
                    draft = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                cb(step, draft)
            if new_api:
                return torch.full((input_ids.shape[0],), flag, dtype=torch.bool, device=input_ids.device)
            return flag
    StopFlag = _StopFlag
    try:
        from wordcloud import WordCloud as _WC
        WordCloud = _WC
    except Exception:
        WordCloud = None            # the app still works, only the word cloud is unavailable
    libs_ready = True


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def cleanText(doc):
    """NLTK preprocessing shown by the 'Preprocess Text' module."""
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in doc.split()]
    tokens = [w for w in tokens if w.lower() not in stop_words]
    tokens = [w for w in tokens if len(w) > 1]
    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return re.sub(r'\d+', '', ' '.join(tokens))


def fix_case(text):
    """Model output can be lowercase: capitalise sentences and end with a full stop."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return text
    text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r'\bi\b', 'I', text)
    if text[-1] not in '.!?':
        text += '.'
    return text


def sentences_of(text):
    """Sentence splitter: NLTK if it works, otherwise a simple rule."""
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return [p for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]


def split_into_chunks(text, count_tokens, limit=CHUNK_TOKENS):
    """Group whole sentences into chunks that stay under 'limit' tokens."""
    chunks, current, size = [], [], 0
    for sent in sentences_of(text):
        n = count_tokens(sent)
        if current and size + n > limit:
            chunks.append(' '.join(current))
            current, size = [], 0
        current.append(sent)
        size += n
    if current:
        chunks.append(' '.join(current))
    return chunks


CLOUD_COLORS = ['#6D6DFF', '#16A34A', '#E5484D', '#D97706', '#0EA5E9', '#A855F7']   # readable on light and dark


def cloud_color(*args, **kwargs):
    return random.choice(CLOUD_COLORS)


def make_cloud_image(text):
    """Word cloud as a PIL image. Transparent background, so it fits both light and dark mode."""
    if WordCloud is None:
        raise RuntimeError('The wordcloud package is missing. Run: pip install wordcloud')
    wc = WordCloud(width=800, height=400, mode='RGBA', background_color=None,
                   stopwords=stop_words, max_words=80, collocations=False,
                   color_func=cloud_color, random_state=1).generate(text)
    return wc.to_image()


def read_document(path):
    """Text from a .txt, .pdf or .docx file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        from pypdf import PdfReader
        return '\n'.join((pg.extract_text() or '') for pg in PdfReader(path).pages)
    if ext == '.docx':
        from docx import Document
        return '\n'.join(par.text for par in Document(path).paragraphs)
    if ext == '.txt':
        with open(path, encoding='utf-8', errors='replace') as f:
            return f.read()
    raise ValueError('Please choose a .pdf, .docx or .txt file.')


def flatten(img):
    """RGBA picture -> RGB on a white background."""
    flat = Image.new('RGB', img.size, 'white')
    flat.paste(img, mask=img.split()[3])
    return flat


def png_stream(img):
    buf = io.BytesIO()
    flatten(img).save(buf, 'PNG')
    buf.seek(0)
    return buf


def export_file(path, summary, stats, cloud_img=None):
    """Save the summary as .txt, .docx or .pdf (the word cloud is added when there is one)."""
    ext = os.path.splitext(path)[1].lower()
    title = 'Abstractive Summary'
    stamp = datetime.now().strftime('%d %b %Y, %H:%M')
    info = f'{stamp}  |  {stats}' if stats else stamp
    if ext == '.txt':
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'{title}\n{info}\n\n{summary}\n')
    elif ext == '.docx':
        from docx import Document
        from docx.shared import Inches
        d = Document()
        d.add_heading(title, 0)
        d.add_paragraph(info)
        d.add_paragraph(summary)
        if cloud_img is not None:
            d.add_heading('Word Cloud', 1)
            d.add_picture(png_stream(cloud_img), width=Inches(5.5))
        d.save(path)
    elif ext == '.pdf':
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
        from xml.sax.saxutils import escape

        def safe(t):    # the built-in PDF fonts only know Latin-1 characters
            for a, b in (('\u2192', '->'), ('\u00b7', '|'), ('\u2018', "'"), ('\u2019', "'"),
                         ('\u201c', '"'), ('\u201d', '"'), ('\u2013', '-'), ('\u2014', '-')):
                t = t.replace(a, b)
            return escape(t.encode('latin-1', 'replace').decode('latin-1'))
        st = getSampleStyleSheet()
        story = [Paragraph(title, st['Title']), Paragraph(safe(info), st['Normal']), Spacer(1, 12),
                 Paragraph(safe(summary), st['BodyText'])]
        if cloud_img is not None:
            story += [Spacer(1, 18), Paragraph('Word Cloud', st['Heading2']),
                      RLImage(png_stream(cloud_img), width=400, height=200)]
        SimpleDocTemplate(path, pagesize=A4, title=title).build(story)
    else:
        raise ValueError('Use .pdf, .docx or .txt')


def length_limits(n_tokens, ratio):
    """Summary length = slider percentage of the input length (never below 5%)."""
    ratio = min(max(ratio, 0.05), 1.0)
    max_len = max(12, int(n_tokens * ratio))
    min_len = max(5, int(max_len * 0.7))
    return min_len, max_len


# ----------------------------------------------------------------------------
# Model functions
# ----------------------------------------------------------------------------
def summarize_piece(piece, beams, ratio, ctx, think):
    global step_callback
    if cancel_event.is_set():
        raise Cancelled()
    prefix = 'summarize: ' if model_name.startswith('t5') else ''
    ids = tokenizer.encode(prefix + piece, return_tensors='pt',
                           max_length=MODEL_LIMIT, truncation=True).to(device)
    min_len, max_len = length_limits(ids.shape[1], ratio)
    think(f'Encoder reads {ids.shape[1]} tokens. Decoder now writes the summary '
          f'({beams} beam{"s" if beams > 1 else ""}, {min_len} to {max_len} tokens).')
    ctx['max'] = max_len
    step_callback = lambda step, draft: ui_queue.put(lambda: app.update_step(ctx, step, draft))
    try:
        with torch.inference_mode():
            out = model.generate(ids,
                                 min_length=min_len,
                                 max_length=max_len,
                                 num_beams=beams,
                                 no_repeat_ngram_size=3,    # stops repeated phrases
                                 length_penalty=2.0,        # prefers fuller summaries
                                 early_stopping=True,
                                 stopping_criteria=[StopFlag()])
    finally:
        step_callback = None
    if cancel_event.is_set():
        raise Cancelled()
    return tokenizer.decode(out[0], skip_special_tokens=True)


def gemini_summary(text, ratio, think):
    n_words = len(text.split())
    target = max(10, round(n_words * min(max(ratio, 0.05), 1.0)))
    think(f'Read the input: {n_words} words. Asking Gemini ({GEMINI_MODEL}) for about {target} words. '
          'Your text is sent to Google for this.')
    if cancel_event.is_set():
        raise Cancelled()
    prompt = (f'Write an abstractive summary of the text below in about {target} words. Use your own words, '
              'plain prose, no bullet points, no title. Return only the summary.\n\n' + text)
    resp = model.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    if cancel_event.is_set():
        raise Cancelled()
    out = (resp.text or '').strip()
    if not out:
        raise RuntimeError('Gemini returned no text (the answer was empty or blocked).')
    return fix_case(out)


def make_summary(text, beams, ratio, think=lambda msg: None):
    if model_name == GEMINI_CHOICE:
        return gemini_summary(text, ratio, think)
    count = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    n_tok = count(text)
    think(f'Read the input: {len(text.split())} words, {n_tok} tokens.')
    if n_tok < 40:
        think('The input is already very short, so it is returned as it is.')
        return text
    chunks = split_into_chunks(text, count)
    ctx = {'part': 1, 'parts': len(chunks), 'max': 100}
    if len(chunks) > 1:
        think(f'The text is long, so it is split into {len(chunks)} parts (up to {CHUNK_TOKENS} tokens each).')
    else:
        think('The text fits in one part.')
    partial = []
    for i, c in enumerate(chunks, 1):
        ctx['part'] = i
        t0 = time.time()
        think(f'Part {i} of {len(chunks)} started.')
        piece = summarize_piece(c, beams, ratio, ctx, think)
        partial.append(piece)
        think(f'Part {i} finished in {time.time() - t0:.1f} s: {len(piece.split())} words written.')
    result = ' '.join(partial)
    think('Fixing capital letters and full stops.')
    return fix_case(result)


# ----------------------------------------------------------------------------
# Modern window (customtkinter = Python only, no HTML/JS)
# ----------------------------------------------------------------------------
SAMPLE = ("Investment banking firms spend large amounts of money acquiring information to drive "
          "their decision-making, including automated stock trading. Financial analysts inevitably "
          "hit a wall and are not able to read everything. Summarisation systems tailored to "
          "financial documents like earning reports and financial news can help analysts quickly "
          "derive market signals from content.")

# (light mode, dark mode) colour pairs
BG      = ("#F3F5FB", "#0B1220")
HEADER  = ("#FFFFFF", "#111A2E")
CARD    = ("#FFFFFF", "#162033")
FIELD   = ("#F7F8FC", "#0F1828")
BORDER  = ("#DDE2EE", "#26334D")
TEXT    = ("#1B2236", "#E8ECF6")
MUTED   = ("#6B7690", "#8D9ABA")
ACCENT  = ("#5B5BF0", "#6D6DFF")
ACCENT_H= ("#4747D6", "#5A5AEA")
SOFT    = ("#E8EAFE", "#22305A")
SOFT_H  = ("#D8DBFB", "#2C3C6E")
GOOD    = "#16A34A"
BAD     = "#E5484D"
WARN    = "#D97706"
FONT    = "Segoe UI"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.title('Abstractive Text Summarization')
        self.geometry('1240x800')
        self.minsize(1020, 700)
        self.configure(fg_color=BG)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.build_header()
        self.build_body()
        self.build_statusbar()
        self.after(50, self.poll)

    def poll(self):
        """Runs functions queued by background threads (safe way to update the window)."""
        try:
            while True:
                ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self.after(50, self.poll)

    # ---------------- header ----------------
    def build_header(self):
        bar = ctk.CTkFrame(self, fg_color=HEADER, corner_radius=0, height=76)
        bar.grid(row=0, column=0, sticky='ew')
        bar.grid_columnconfigure(1, weight=1)
        badge = ctk.CTkLabel(bar, text='AS', width=44, height=44, corner_radius=12,
                             fg_color=ACCENT, text_color='white', font=(FONT, 17, 'bold'))
        badge.grid(row=0, column=0, rowspan=2, padx=(26, 14), pady=16)
        ctk.CTkLabel(bar, text='Abstractive Text Summarization', text_color=TEXT,
                     font=(FONT, 22, 'bold'), anchor='w').grid(row=0, column=1, sticky='sw', pady=(14, 0))
        ctk.CTkLabel(bar, text='Writes new, short summaries with a Transformer (T5)', text_color=MUTED,
                     font=(FONT, 13), anchor='w').grid(row=1, column=1, sticky='nw', pady=(0, 14))
        self.pill = ctk.CTkLabel(bar, text='●  Starting up ...', height=32, corner_radius=16,
                                 fg_color=SOFT, text_color=MUTED, font=(FONT, 13, 'bold'), padx=14)
        self.pill.grid(row=0, column=2, rowspan=2, padx=10)
        self.mode = ctk.CTkSwitch(bar, text='Dark', command=self.toggle_mode, font=(FONT, 13),
                                  text_color=MUTED, progress_color=ACCENT, width=70)
        self.mode.select()
        self.mode.grid(row=0, column=3, rowspan=2, padx=(6, 26))

    def toggle_mode(self):
        ctk.set_appearance_mode('dark' if self.mode.get() else 'light')

    # ---------------- body ----------------
    def build_body(self):
        body = ctk.CTkFrame(self, fg_color='transparent')
        body.grid(row=1, column=0, sticky='nsew', padx=24, pady=(20, 8))
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self.build_sidebar(body)
        self.build_panels(body)

    def card(self, parent):
        return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER)

    def build_sidebar(self, parent):
        side = self.card(parent)
        side.grid(row=0, column=0, sticky='ns', padx=(0, 20))
        side.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(side, text='Workflow', text_color=TEXT, font=(FONT, 17, 'bold'),
                     anchor='w').grid(row=0, column=0, sticky='ew', padx=22, pady=(20, 4))
        ctk.CTkLabel(side, text='Follow the steps from top to bottom', text_color=MUTED,
                     font=(FONT, 12), anchor='w').grid(row=1, column=0, sticky='ew', padx=22, pady=(0, 14))

        self.step(side, 2, '1', 'Load the model')
        self.chosen_model = ctk.StringVar(value=MODEL_CHOICES[0])
        self.model_menu = ctk.CTkOptionMenu(side, values=MODEL_CHOICES, variable=self.chosen_model, height=36,
                          fg_color=FIELD, button_color=SOFT, button_hover_color=SOFT_H,
                          text_color=TEXT, dropdown_fg_color=CARD, dropdown_text_color=TEXT,
                          font=(FONT, 13), dropdown_font=(FONT, 13), corner_radius=10,
                          command=lambda v: refresh())
        self.model_menu.grid(row=3, column=0, sticky='ew', padx=22, pady=(0, 8))
        self.btn_load = ctk.CTkButton(side, text='Generate & Load Transformer Model', height=42,
                                      corner_radius=10, fg_color=ACCENT, hover_color=ACCENT_H,
                                      text_color='white', font=(FONT, 13, 'bold'), command=loadModel_ui)
        self.btn_load.grid(row=4, column=0, sticky='ew', padx=22, pady=(0, 14))

        self.step(side, 5, '2', 'Clean the text (optional)')
        self.btn_pre = ctk.CTkButton(side, text='Preprocess Text', height=40, corner_radius=10,
                                     fg_color=SOFT, hover_color=SOFT_H, text_color=TEXT,
                                     font=(FONT, 13, 'bold'), command=preprocess_ui)
        self.btn_pre.grid(row=6, column=0, sticky='ew', padx=22, pady=(0, 14))

        self.step(side, 7, '3', 'Create the summary')
        opt = ctk.CTkFrame(side, fg_color='transparent')
        opt.grid(row=8, column=0, sticky='ew', padx=22, pady=(0, 8))
        opt.grid_columnconfigure(0, weight=1)
        self.length_lbl = ctk.CTkLabel(opt, text='', text_color=TEXT, font=(FONT, 12), anchor='w')
        self.length_lbl.grid(row=0, column=0, sticky='ew')
        self.length_var = ctk.DoubleVar(value=30)
        self.length_slider = ctk.CTkSlider(opt, from_=0, to=100, number_of_steps=20, variable=self.length_var,
                                           progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_H,
                                           command=lambda v: self.update_length_label())
        self.length_slider.grid(row=1, column=0, sticky='ew', pady=(2, 6))
        self.speed_var = ctk.StringVar(value='Balanced (2 beams)')
        self.speed_menu = ctk.CTkOptionMenu(opt, values=list(SPEED_CHOICES), variable=self.speed_var, height=32,
                          fg_color=FIELD, button_color=SOFT, button_hover_color=SOFT_H, text_color=TEXT,
                          dropdown_fg_color=CARD, dropdown_text_color=TEXT, font=(FONT, 12),
                          dropdown_font=(FONT, 12), corner_radius=10)
        self.speed_menu.grid(row=2, column=0, sticky='ew')
        self.update_length_label()
        self.btn_sum = ctk.CTkButton(side, text='Generate Abstractive Summary', height=44, corner_radius=10,
                                     fg_color=ACCENT, hover_color=ACCENT_H, text_color='white',
                                     font=(FONT, 14, 'bold'), command=summary_ui)
        self.btn_sum.grid(row=9, column=0, sticky='ew', padx=22, pady=(0, 8))
        self.btn_stop = ctk.CTkButton(side, text='■  Stop', height=36, corner_radius=10, fg_color=BAD,
                                      hover_color='#C93A3F', text_color='white', font=(FONT, 13, 'bold'),
                                      command=stop_ui, state='disabled')
        self.btn_stop.grid(row=10, column=0, sticky='ew', padx=22, pady=(0, 14))

        ctk.CTkFrame(side, height=1, fg_color=BORDER).grid(row=11, column=0, sticky='ew', padx=22, pady=(0, 12))
        row = ctk.CTkFrame(side, fg_color='transparent')
        row.grid(row=12, column=0, sticky='ew', padx=22, pady=(0, 8))
        row.grid_columnconfigure((0, 1), weight=1, uniform='b')
        self.btn_sample = ctk.CTkButton(row, text='Sample Text', height=34, corner_radius=10, fg_color='transparent',
                      border_width=1, border_color=BORDER, hover_color=SOFT, text_color=TEXT,
                      font=(FONT, 12), command=insert_sample)
        self.btn_sample.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self.btn_clear = ctk.CTkButton(row, text='Clear Text', height=34, corner_radius=10, fg_color='transparent',
                      border_width=1, border_color=BORDER, hover_color=SOFT, text_color=BAD,
                      font=(FONT, 12, 'bold'), command=clearText_ui)
        self.btn_clear.grid(row=0, column=1, sticky='ew', padx=(4, 0))
        self.device_lbl = ctk.CTkLabel(side, text='Starting up ...', text_color=MUTED,
                                       font=(FONT, 12), anchor='w')
        self.device_lbl.grid(row=13, column=0, sticky='ew', padx=22, pady=(0, 16))
        side.grid_rowconfigure(13, weight=1)

    def step(self, parent, row, num, label):
        f = ctk.CTkFrame(parent, fg_color='transparent')
        f.grid(row=row, column=0, sticky='ew', padx=22, pady=(0, 8))
        ctk.CTkLabel(f, text=num, width=24, height=24, corner_radius=12, fg_color=SOFT, text_color=ACCENT,
                     font=(FONT, 12, 'bold')).pack(side='left')
        ctk.CTkLabel(f, text=label, text_color=TEXT, font=(FONT, 13, 'bold')).pack(side='left', padx=10)

    def build_panels(self, parent):
        right = ctk.CTkFrame(parent, fg_color='transparent')
        right.grid(row=0, column=1, sticky='nsew')
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure((0, 1), weight=1, uniform='p')
        # input card
        c1 = self.card(right)
        c1.grid(row=0, column=0, sticky='nsew', pady=(0, 14))
        c1.grid_columnconfigure(0, weight=1); c1.grid_rowconfigure(1, weight=1)
        h1 = ctk.CTkFrame(c1, fg_color='transparent'); h1.grid(row=0, column=0, sticky='ew', padx=22, pady=(16, 6))
        ctk.CTkLabel(h1, text='Input Your Text Below', text_color=TEXT, font=(FONT, 16, 'bold')).pack(side='left')
        self.in_count = ctk.CTkLabel(h1, text='0 words', text_color=MUTED, font=(FONT, 12))
        self.in_count.pack(side='right')
        self.btn_open = ctk.CTkButton(h1, text='Open file (PDF / DOCX / TXT)', width=190, height=26, corner_radius=8,
                                      fg_color=SOFT, hover_color=SOFT_H, text_color=TEXT, font=(FONT, 12, 'bold'),
                                      command=open_file_ui)
        self.btn_open.pack(side='right', padx=(0, 14))
        self.inputtext = ctk.CTkTextbox(c1, wrap='word', font=(FONT, 15), corner_radius=10, fg_color=FIELD,
                                        text_color=TEXT, border_width=1, border_color=BORDER)
        self.inputtext.grid(row=1, column=0, sticky='nsew', padx=22, pady=(0, 18))
        self.inputtext.bind('<KeyRelease>', lambda e: self.update_in_count())
        # output card
        c2 = self.card(right)
        c2.grid(row=1, column=0, sticky='nsew')
        c2.grid_columnconfigure(0, weight=1); c2.grid_rowconfigure(1, weight=1)
        h2 = ctk.CTkFrame(c2, fg_color='transparent'); h2.grid(row=0, column=0, sticky='ew', padx=22, pady=(16, 6))
        self.out_title = ctk.CTkLabel(h2, text='Output', text_color=TEXT, font=(FONT, 16, 'bold'))
        self.out_title.pack(side='left')
        ctk.CTkButton(h2, text='Copy', width=70, height=28, corner_radius=8, fg_color=SOFT, hover_color=SOFT_H,
                      text_color=TEXT, font=(FONT, 12, 'bold'), command=self.copy_out).pack(side='right')
        self.view_btn = ctk.CTkButton(h2, text='Show thinking', width=110, height=28, corner_radius=8,
                      fg_color='transparent', border_width=1, border_color=BORDER, hover_color=SOFT,
                      text_color=TEXT, font=(FONT, 12), state='disabled', command=self.toggle_view)
        self.view_btn.pack(side='right', padx=(0, 8))
        self.btn_export = ctk.CTkButton(h2, text='Export', width=70, height=28, corner_radius=8, fg_color=SOFT,
                                        hover_color=SOFT_H, text_color=TEXT, font=(FONT, 12, 'bold'),
                                        state='disabled', command=self.export_summary)
        self.btn_export.pack(side='right', padx=(0, 8))
        self.last_summary = self.last_log = None
        self.last_stats = ''
        self.thinking = False
        self.out_stats = ctk.CTkLabel(h2, text='', text_color=MUTED, font=(FONT, 12))
        self.out_stats.pack(side='right', padx=14)
        self.last_input = ''
        self.cloud_img, self.cloud_size, self.cloud_token = None, None, 0
        box = ctk.CTkFrame(c2, fg_color='transparent')
        box.grid(row=1, column=0, sticky='nsew', padx=22, pady=(0, 18))
        box.grid_columnconfigure(0, weight=3, uniform='o'); box.grid_columnconfigure(1, weight=2, uniform='o')
        box.grid_rowconfigure(0, weight=1)
        self.text = ctk.CTkTextbox(box, wrap='word', font=(FONT, 15), corner_radius=10, fg_color=FIELD,
                                   text_color=TEXT, border_width=1, border_color=BORDER)
        self.text.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.build_cloud(box)

    # ---------------- Word cloud panel ----------------
    def build_cloud(self, parent):
        cloud = ctk.CTkFrame(parent, fg_color=FIELD, corner_radius=10, border_width=1, border_color=BORDER)
        cloud.grid(row=0, column=1, sticky='nsew')
        cloud.grid_columnconfigure(0, weight=1); cloud.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(cloud, fg_color='transparent')
        top.grid(row=0, column=0, sticky='ew', padx=10, pady=(8, 0))
        ctk.CTkLabel(top, text='Word Cloud', text_color=TEXT, font=(FONT, 13, 'bold')).pack(side='left')
        self.btn_cloud_save = ctk.CTkButton(top, text='Save', width=52, height=24, corner_radius=8, fg_color=SOFT,
                                            hover_color=SOFT_H, text_color=TEXT, font=(FONT, 11, 'bold'),
                                            state='disabled', command=self.save_cloud)
        self.btn_cloud_save.pack(side='right')
        self.btn_cloud_big = ctk.CTkButton(top, text='Enlarge', width=60, height=24, corner_radius=8, fg_color=SOFT,
                                           hover_color=SOFT_H, text_color=TEXT, font=(FONT, 11, 'bold'),
                                           state='disabled', command=self.enlarge_cloud)
        self.btn_cloud_big.pack(side='right', padx=(0, 6))
        # fixed-size area (no propagation) so the picture can be scaled to whatever space is available
        self.cloud_area = ctk.CTkFrame(cloud, fg_color='transparent')
        self.cloud_area.grid(row=1, column=0, sticky='nsew', padx=6, pady=6)
        self.cloud_area.grid_propagate(False)
        self.cloud_area.bind('<Configure>', lambda e: self.fit_cloud())
        self.cloud_lbl = ctk.CTkLabel(self.cloud_area, text='')
        self.cloud_msg = ctk.CTkLabel(self.cloud_area, text='', text_color=MUTED, font=(FONT, 12), wraplength=230)
        self.cloud_src = ctk.StringVar(value='Original text')
        self.cloud_menu = ctk.CTkOptionMenu(cloud, values=['Original text', 'Summary'], variable=self.cloud_src,
                          height=26, fg_color=CARD, button_color=SOFT, button_hover_color=SOFT_H, text_color=TEXT,
                          dropdown_fg_color=CARD, dropdown_text_color=TEXT, font=(FONT, 12),
                          dropdown_font=(FONT, 12), corner_radius=8, command=lambda v: self.refresh_cloud())
        self.cloud_menu.grid(row=2, column=0, sticky='ew', padx=10, pady=(0, 10))
        self.set_cloud_message('The word cloud appears here after you create a summary.')

    def set_cloud_message(self, msg):
        self.cloud_img, self.cloud_size = None, None
        self.cloud_lbl.place_forget()
        self.cloud_msg.configure(text=msg)
        self.cloud_msg.place(relx=0.5, rely=0.5, anchor='center')
        self.btn_cloud_save.configure(state='disabled')
        self.btn_cloud_big.configure(state='disabled')

    def fit_cloud(self):
        """Scale the picture to the free space, keeping its proportions."""
        if self.cloud_img is None:
            return
        w, h = self.cloud_area.winfo_width() - 8, self.cloud_area.winfo_height() - 8
        if w < 40 or h < 40:
            return
        iw, ih = self.cloud_img.size
        k = min(w / iw, h / ih)
        size = (max(int(iw * k), 1), max(int(ih * k), 1))
        if size == self.cloud_size:
            return
        self.cloud_size = size
        self.cloud_ctk = ctk.CTkImage(light_image=self.cloud_img, dark_image=self.cloud_img, size=size)
        self.cloud_lbl.configure(image=self.cloud_ctk)
        self.cloud_msg.place_forget()
        self.cloud_lbl.place(relx=0.5, rely=0.5, anchor='center')

    def refresh_cloud(self):
        """Draw the cloud for the chosen source (runs in a thread so the window never freezes)."""
        text = self.last_summary if self.cloud_src.get() == 'Summary' else self.last_input
        if not text:
            self.set_cloud_message('Create a summary first.')
            return
        self.cloud_token += 1
        token = self.cloud_token
        self.set_cloud_message('Drawing word cloud ...')

        def done(result, error):
            if token != self.cloud_token:          # a newer request replaced this one
                return
            if error:
                msg = 'Not enough words to draw a cloud.' if isinstance(error, ValueError) else str(error)
                self.set_cloud_message(msg)
                return
            self.cloud_img, self.cloud_size = result, None
            self.btn_cloud_save.configure(state='normal')
            self.btn_cloud_big.configure(state='normal')
            self.fit_cloud()

        run_in_background(lambda: make_cloud_image(text), done)

    def enlarge_cloud(self):
        if self.cloud_img is None:
            return
        win = ctk.CTkToplevel(self)
        win.title('Word Cloud')
        win.geometry('840x440')
        win.configure(fg_color=FIELD)
        win.big = ctk.CTkImage(light_image=self.cloud_img, dark_image=self.cloud_img, size=self.cloud_img.size)
        ctk.CTkLabel(win, text='', image=win.big).pack(expand=True, padx=16, pady=16)
        win.after(150, win.lift)

    def export_summary(self):
        if not self.last_summary:
            return
        path = filedialog.asksaveasfilename(defaultextension='.pdf', initialfile='summary.pdf',
                                            filetypes=[('PDF document', '*.pdf'), ('Word document', '*.docx'),
                                                       ('Text file', '*.txt')])
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in ('.pdf', '.docx', '.txt'):
            path += '.pdf'
        try:
            export_file(path, self.last_summary, self.last_stats, self.cloud_img)
        except ImportError as e:
            self.set_status('Export needs a package: pip install python-docx reportlab  (' + str(e) + ')')
            return
        except Exception as e:
            self.set_status('Export failed: ' + str(e))
            return
        self.set_status('Exported: ' + path)

    def save_cloud(self):
        if self.cloud_img is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile='word_cloud.png',
                                            filetypes=[('PNG image', '*.png')])
        if not path:
            return
        flatten(self.cloud_img).save(path)       # white background for the saved file
        self.set_status('Word cloud saved: ' + path)

    # ---------------- status bar ----------------
    def build_statusbar(self):
        bar = ctk.CTkFrame(self, fg_color=HEADER, corner_radius=0, height=38)
        bar.grid(row=2, column=0, sticky='ew')
        bar.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(bar, text='Starting up: loading libraries in the background ...', text_color=MUTED,
                                   font=(FONT, 12), anchor='w')
        self.status.grid(row=0, column=0, sticky='ew', padx=26, pady=8)
        self.progress = ctk.CTkProgressBar(bar, width=200, height=6, progress_color=ACCENT, fg_color=BORDER,
                                           mode='indeterminate')
        self.progress.grid(row=0, column=1, padx=26)
        self.progress.grid_remove()

    # ---------------- helpers used by the actions ----------------
    def update_length_label(self):
        v = int(round(self.length_var.get()))
        self.length_lbl.configure(text='Summary length: minimum (5%)' if v < 5
                                  else f'Summary length: {v}% of the original')

    def update_in_count(self):
        n = len(self.inputtext.get('1.0', 'end-1c').split())
        self.in_count.configure(text=f'{n} words')

    def set_status(self, msg, busy=False):
        self.status.configure(text=msg)
        if busy:
            self.progress.configure(mode='indeterminate')
            self.progress.grid(); self.progress.start()
        else:
            self.progress.stop(); self.progress.grid_remove()

    # ---------------- Thinking panel ----------------
    def start_thinking(self):
        self.think_lines, self.draft, self.step_text = [], '', ''
        self.thinking, self.t_start = True, time.time()
        self.last_summary = self.last_log = None
        self.view_btn.configure(state='disabled', text='Show thinking')
        self.out_title.configure(text='Thinking ...')
        self.out_stats.configure(text='')
        self.progress.stop(); self.progress.configure(mode='determinate'); self.progress.set(0); self.progress.grid()
        self.render_think()
        self.after(200, self.tick)

    def tick(self):
        if self.thinking:
            self.render_think()
            self.after(200, self.tick)

    def add_think(self, msg):
        self.think_lines.append(msg)
        self.status.configure(text=msg)
        self.render_think()

    def update_step(self, ctx, step, draft):
        if not self.thinking:
            return
        self.step_text = f'Decoder step {step} of about {ctx["max"]} tokens'
        frac = ((ctx['part'] - 1) + min(step / max(ctx['max'], 1), 1.0)) / max(ctx['parts'], 1)
        self.progress.set(min(frac, 1.0))
        if draft:
            self.draft = draft
        self.render_think()

    def think_text(self, final=False):
        lines = [('✓  ' if (final or i < len(self.think_lines) - 1) else '→  ') + m
                 for i, m in enumerate(self.think_lines)]
        if self.step_text and not final:
            lines.append('    ' + self.step_text)
        if self.draft and not final:
            lines += ['', 'Draft so far (the decoder may still change it):', self.draft]
        return '\n'.join(lines)

    def render_think(self):
        el = time.time() - self.t_start
        self.out_stats.configure(text=f'{el:.1f} s')
        self.text.delete('1.0', 'end')
        self.text.insert('end', self.think_text())

    def end_thinking(self):
        self.thinking = False
        el = time.time() - self.t_start
        self.last_log = self.think_text(final=True) + f'\n\nTotal time: {el:.1f} s'
        self.view_btn.configure(state='normal')

    def toggle_view(self):
        if self.view_btn.cget('text') == 'Show thinking':
            self.show(self.last_log, 'Thinking (steps taken)')
            self.view_btn.configure(text='Show summary')
        else:
            self.show(self.last_summary or '', 'Abstractive Summary', self.last_stats)
            self.view_btn.configure(text='Show thinking')

    def show(self, message, title='Output', stats=''):
        self.out_title.configure(text=title)
        self.out_stats.configure(text=stats)
        self.text.delete('1.0', 'end')
        self.text.insert('end', message)

    def copy_out(self):
        self.clipboard_clear()
        self.clipboard_append(self.text.get('1.0', 'end-1c'))
        self.set_status('Copied to clipboard.')

    def set_pill(self, text, color):
        self.pill.configure(text='●  ' + text, text_color=color)


# ----------------------------------------------------------------------------
# Actions (the three project modules)
# ----------------------------------------------------------------------------
busy = False
cancellable = False


def refresh():
    """One place that decides which controls are usable."""
    sel = app.chosen_model.get()
    idle = not busy
    app.model_menu.configure(state='normal' if idle else 'disabled')      # no model change while working
    app.speed_menu.configure(state='normal' if idle else 'disabled')
    app.btn_load.configure(state='normal' if (idle and libs_ready) else 'disabled')
    app.btn_pre.configure(state='normal' if (idle and libs_ready) else 'disabled')
    app.btn_clear.configure(state='normal' if idle else 'disabled')
    app.btn_sample.configure(state='normal' if idle else 'disabled')
    app.btn_open.configure(state='normal' if idle else 'disabled')
    ready = idle and libs_ready and model is not None and model_name == sel
    app.btn_sum.configure(state='normal' if ready else 'disabled')
    app.btn_stop.configure(state='normal' if (busy and cancellable) else 'disabled')
    if busy:
        return
    if not libs_ready:
        app.set_pill('Starting up ...', MUTED)
    elif model is None:
        app.set_pill('Model not loaded', BAD)
    elif model_name == sel:
        app.set_pill('Transformer Model Loaded (' + model_name + ')', GOOD)
    else:
        app.set_pill('Selected ' + sel + ': click Generate & Load', WARN)


def run_in_background(work, done):
    """Slow work runs in a thread so the window never freezes."""
    def target():
        try:
            result, error = work(), None
        except BaseException as e:
            result, error = None, e
        ui_queue.put(lambda: done(result, error))
    threading.Thread(target=target, daemon=True).start()


def start_task(status, pill_text, can_stop=True):
    global busy, cancellable
    busy, cancellable = True, can_stop
    cancel_event.clear()
    app.set_status(status, busy=True)
    app.set_pill(pill_text, MUTED)
    refresh()


def end_task():
    global busy, cancellable
    busy, cancellable = False, False
    app.progress.stop(); app.progress.configure(mode='indeterminate'); app.progress.grid_remove()
    refresh()


def start_up_done(result, error):
    if error:
        app.set_status('Start-up failed: ' + str(error))
        app.show('Could not start the libraries:\n' + str(error) + '\n\nMost likely torch or transformers is missing. Run: pip install -r requirements.txt', 'Error')
        app.set_pill('Start-up failed', BAD)
        app.progress.stop(); app.progress.grid_remove()
        return
    app.device_lbl.configure(text='Running on: ' + str(device).upper())
    app.set_status('Ready. Step 1: choose a model and load it.')
    end_task()


def loadModel_ui():
    global gemini_key
    name = app.chosen_model.get()
    if name == GEMINI_CHOICE and not gemini_key:
        gemini_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY') or ''
        if not gemini_key:
            dlg = ctk.CTkInputDialog(text='Paste your Gemini API key (kept in memory only).\n'
                                          'Note: your text is sent to Google when you use Gemini.',
                                     title='Gemini API key')
            gemini_key = (dlg.get_input() or '').strip()
        if not gemini_key:
            app.set_status('Gemini needs an API key.')
            return
    start_task('Loading ' + name + '. The first time it is downloaded, so please wait ...', 'Loading ' + name)

    def work():
        if name == GEMINI_CHOICE:
            from google import genai                  # pip install google-genai
            client = genai.Client(api_key=gemini_key)
            client.models.generate_content(model=GEMINI_MODEL, contents='Reply with OK.')   # checks key + model
            return None, client
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
        mod.eval()
        return tok, mod

    def done(result, error):
        global model, tokenizer, model_name, gemini_key
        if cancel_event.is_set():
            app.set_status('Loading cancelled.')
        elif error:
            app.set_status('Could not load the model.')
            if name == GEMINI_CHOICE:
                gemini_key = ''                   # ask for the key again next time
            app.show('Could not load the model:\n' + str(error), 'Error')
        else:
            tokenizer, model = result
            model_name = name
            app.set_status('Gemini ready. Note: your text is sent to Google.' if name == GEMINI_CHOICE
                           else 'Model loaded. Enter text and create a summary.')
            app.show('Model loaded', 'Output')
        app.progress.stop(); app.progress.grid_remove()
        end_task()

    run_in_background(work, done)


def preprocess_ui():
    data = app.inputtext.get('1.0', 'end-1c').strip()
    if not data:
        app.set_status('Please enter some text first.')
        return
    app.show(cleanText(data), 'Preprocessed Text', 'stop words, symbols and numbers removed')
    app.set_status('Text cleaned. This view is only for display; the summary uses your original text.')


def summary_ui():
    data = app.inputtext.get('1.0', 'end-1c').strip().replace('\n', ' ')
    if not data:
        app.set_status('Please enter some text first.')
        return
    beams = SPEED_CHOICES[app.speed_var.get()]
    ratio = app.length_var.get() / 100
    start_task('Generating summary ...', 'Generating ...')
    app.start_thinking()
    t0 = time.time()
    think = lambda msg: ui_queue.put(lambda: app.add_think(msg))

    def done(result, error):
        app.end_thinking()
        if isinstance(error, Cancelled):
            app.last_log += '\n\nStopped by the user.'
            app.set_status('Stopped.')
            app.show('Stopped. Click "Generate Abstractive Summary" to try again.', 'Output')
            app.view_btn.configure(text='Show thinking')
        elif error:
            app.set_status('Something went wrong.')
            app.show('Error: ' + str(error), 'Error')
        else:
            n_in, n_out = len(data.split()), len(result.split())
            cut = max(0, round(100 * (1 - n_out / max(n_in, 1))))
            secs = time.time() - t0
            app.last_summary = result
            app.last_stats = f'{n_in} → {n_out} words  ·  {100 - cut}% of original  ·  {secs:.1f} s'
            app.show(result, 'Abstractive Summary', app.last_stats)
            app.view_btn.configure(text='Show thinking')
            app.set_status(f'Summary ready in {secs:.1f} seconds.')
            app.last_input = data
            app.btn_export.configure(state='normal')
            app.refresh_cloud()
        end_task()

    run_in_background(lambda: make_summary(data, beams, ratio, think), done)


def stop_ui():
    cancel_event.set()
    app.btn_stop.configure(state='disabled')
    if app.thinking:
        app.add_think('Stop requested ...')
    else:
        app.set_status('Stopping ... (a model download cannot be interrupted, it will stop when it ends)', busy=True)


def open_file_ui():
    path = filedialog.askopenfilename(filetypes=[('Documents', '*.pdf *.docx *.txt'), ('All files', '*.*')])
    if not path:
        return
    name = os.path.basename(path)
    start_task('Reading ' + name + ' ...', 'Reading file', can_stop=False)

    def done(result, error):
        if error:
            app.set_status('Could not read the file: ' + str(error))
        elif not (result or '').strip():
            app.set_status('No text found in ' + name + ' (scanned PDFs need OCR).')
        else:
            app.inputtext.delete('1.0', 'end')
            app.inputtext.insert('end', result.strip())
            app.update_in_count()
            app.set_status(f'Loaded {name} ({len(result.split())} words).')
        end_task()

    run_in_background(lambda: read_document(path), done)


def insert_sample():
    app.inputtext.delete('1.0', 'end')
    app.inputtext.insert('end', SAMPLE)
    app.update_in_count()
    app.set_status('Sample text inserted.')


def clearText_ui():
    app.text.delete('1.0', 'end')
    app.inputtext.delete('1.0', 'end')
    app.out_title.configure(text='Output')
    app.out_stats.configure(text='')
    app.last_input, app.last_summary = '', None
    app.btn_export.configure(state='disabled')
    app.cloud_token += 1
    app.set_cloud_message('The word cloud appears here after you create a summary.')
    app.update_in_count()
    app.set_status('Cleared.')


if __name__ == '__main__':
    app = App()
    busy, cancellable = True, False          # disabled until the libraries are ready
    refresh()
    app.set_status('Starting up: loading libraries in the background ...', busy=True)
    run_in_background(init_libs, start_up_done)
    app.mainloop()