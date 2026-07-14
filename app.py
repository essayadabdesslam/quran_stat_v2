# -*- coding: utf-8 -*-
"""
تطبيق تحليل لغوي إحصائي للنص القرآني الكامل (114 سورة، 6236 آية)
Application Streamlit (interface en arabe) — analyse lexicale/statistique
du texte coranique complet, avec sélection de sourates et export PDF.

تنبيه منهجي مهم
-----------------
هذه الأداة أداة وصفية بحتة (إحصاءات معجمية، تناص لغوي سطحي عبر TF-IDF).
إنها لا تنتج أي تفسير ديني أو حكم شرعي، ولا يجوز اعتبار نتائجها بديلاً
عن التفسير المعتمد على علم أصيل. أي استخدام تأويلي لنتائج هذه الأداة
يجب أن يخضع لمراجعة أهل الاختصاص (علماء، مفسرون، لغويون).

Lancement local :
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import arabic_reshaper
import pandas as pd
import pyarabic.araby as araby
import streamlit as st
from bidi.algorithm import get_display
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                 Spacer, Table, TableStyle)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "quran_data.json"
FONT_REGULAR = BASE_DIR / "fonts" / "Amiri-Regular.ttf"
FONT_BOLD = BASE_DIR / "fonts" / "Amiri-Bold.ttf"

st.set_page_config(page_title="تحليل لغوي للقرآن الكريم", page_icon="📖", layout="wide")

# ---------------------------------------------------------------------------
# تنسيق الاتجاه من اليمين إلى اليسار (RTL) لكامل الواجهة
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox,
    .stMultiSelect, .stButton, .stDataFrame, .stTabs, .stRadio, .stSlider {
        direction: rtl;
        text-align: right;
        font-family: "Traditional Arabic", "Arial", sans-serif;
    }
    [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    .stTextInput input { text-align: right; direction: rtl; }
    .quran-text { font-size: 22px; line-height: 2.4; text-align: right; direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# تحميل البيانات والمعالجة الأولية
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="جارٍ تحميل النص القرآني الكامل...")
def charger_corpus():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def normaliser(texte: str) -> str:
    texte = araby.strip_diacritics(texte)
    texte = araby.normalize_hamza(texte)
    texte = araby.normalize_ligature(texte)
    texte = araby.strip_tatweel(texte)
    return texte


@st.cache_data(show_spinner="جارٍ تجهيز فهرس الآيات...")
def construire_index(_corpus):
    lignes = []
    for surah in _corpus:
        for verse in surah["verses"]:
            lignes.append({
                "surah_id": surah["id"],
                "surah_name": surah["name"],
                "surah_translit": surah["transliteration"],
                "surah_type": surah["type"],
                "ayah": verse["id"],
                "text": verse["text"],
                "text_norm": normaliser(verse["text"]),
                "ref": f'{surah["id"]}:{verse["id"]}',
            })
    df = pd.DataFrame(lignes)
    df["n_words"] = df["text_norm"].apply(lambda t: len(araby.tokenize(t)))
    return df


@st.cache_resource(show_spinner="جارٍ بناء نموذج التشابه اللغوي (TF-IDF)...")
def construire_vectoriseur(textes_norm: tuple):
    vect = TfidfVectorizer(analyzer="word", token_pattern=r"[^\s]+")
    matrice = vect.fit_transform(textes_norm)
    return vect, matrice


corpus = charger_corpus()
index_df = construire_index(corpus)
vectoriseur, matrice_tfidf = construire_vectoriseur(tuple(index_df["text_norm"]))

options_sourates = [
    f'{s["id"]} - {s["name"]} ({s["transliteration"]})' for s in corpus
]
id_depuis_option = {opt: int(opt.split(" - ")[0]) for opt in options_sourates}

# ---------------------------------------------------------------------------
# الشريط الجانبي — تحديد نطاق التحليل
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ نطاق التحليل")
نطاق = st.sidebar.radio("اختر النطاق:", ["📖 القرآن الكريم كاملاً", "📑 اختيار سور محددة"])

if نطاق == "📖 القرآن الكريم كاملاً":
    سور_مختارة = [s["id"] for s in corpus]
else:
    اختيار = st.sidebar.multiselect(
        "اختر سورة أو أكثر:", options_sourates, default=[options_sourates[0]]
    )
    سور_مختارة = [id_depuis_option[o] for o in اختيار] or [1]

scope_df = index_df[index_df["surah_id"].isin(سور_مختارة)].reset_index(drop=True)

st.sidebar.markdown("---")
st.sidebar.metric("عدد السور المختارة", len(سور_مختارة))
st.sidebar.metric("عدد الآيات في النطاق", len(scope_df))

# ---------------------------------------------------------------------------
# الرأس والتنبيه المنهجي
# ---------------------------------------------------------------------------

st.title("📖 أداة التحليل اللغوي للقرآن الكريم")
st.caption("نسخة تجريبية · معالجة اللغة العربية (NLP) · pyarabic + scikit-learn")

st.warning(
    "**تنبيه منهجي** — هذه أداة وصفية بحتة (إحصاءات معجمية وتشابه نصي سطحي). "
    "إنها لا تُنتج أي تفسير ديني ولا تُغني بأي حال عن التفسير المعتمد على "
    "علم شرعي أصيل. أي استخدام تأويلي لنتائجها يجب أن يخضع لمراجعة أهل "
    "الاختصاص (علماء، مفسرون، لغويون متخصصون في العربية الفصحى القرآنية)."
)

تبويب_لوحة, تبويب_نص, تبويب_بحث, تبويب_تكرار, تبويب_تشابه, تبويب_تقرير = st.tabs(
    ["🧮 لوحة القيادة", "📚 النص", "🔎 البحث والتوافق", "📊 التكرار المعجمي",
     "🧭 التشابه النصي", "📄 تقرير PDF"]
)

# ---------------------------------------------------------------------------
# تبويب 0: لوحة القيادة — إحصائيات عامة حول كلمات القرآن الكريم
# ---------------------------------------------------------------------------

النوع_عربي = {"meccan": "مكية", "medinan": "مدنية"}

with تبويب_لوحة:
    st.subheader("لوحة القيادة — إحصائيات كلمات القرآن الكريم")

    نطاق_لوحة = st.radio(
        "نطاق الإحصائيات:",
        ["📖 القرآن الكريم كاملاً", "📑 النطاق المحدد في الشريط الجانبي"],
        horizontal=True, key="نطاق_لوحة",
    )
    df_لوحة = index_df if نطاق_لوحة == "📖 القرآن الكريم كاملاً" else scope_df

    if len(df_لوحة) == 0:
        st.info("لا توجد آيات ضمن هذا النطاق.")
    else:
        كل_كلمات_لوحة = []
        for texte in df_لوحة["text_norm"]:
            كل_كلمات_لوحة.extend(araby.tokenize(texte))

        عدد_السور_لوحة = df_لوحة["surah_id"].nunique()
        عدد_الآيات_لوحة = len(df_لوحة)
        إجمالي_الكلمات = len(كل_كلمات_لوحة)
        كلمات_فريدة = len(set(كل_كلمات_لوحة))
        متوسط_كلمات = إجمالي_الكلمات / عدد_الآيات_لوحة if عدد_الآيات_لوحة else 0
        نسبة_التنوع = (كلمات_فريدة / إجمالي_الكلمات * 100) if إجمالي_الكلمات else 0

        # --- المؤشرات الرئيسية ---
        ع1, ع2, ع3, ع4, ع5 = st.columns(5)
        ع1.metric("عدد السور", f"{عدد_السور_لوحة:,}")
        ع2.metric("عدد الآيات", f"{عدد_الآيات_لوحة:,}")
        ع3.metric("إجمالي الكلمات", f"{إجمالي_الكلمات:,}")
        ع4.metric("الكلمات الفريدة", f"{كلمات_فريدة:,}")
        ع5.metric("متوسط الكلمات/آية", f"{متوسط_كلمات:.1f}")

        st.caption(f"نسبة التنوع المعجمي (كلمات فريدة / إجمالي الكلمات): {نسبة_التنوع:.1f}٪")
        st.markdown("---")

        # --- أطول وأقصر الآيات ---
        عمود_أ, عمود_ب = st.columns(2)
        with عمود_أ:
            st.markdown("**📏 أطول 10 آيات (بعدد الكلمات)**")
            اطول = df_لوحة.nlargest(10, "n_words")[["surah_name", "ref", "n_words"]].rename(
                columns={"surah_name": "السورة", "ref": "الآية", "n_words": "عدد الكلمات"}
            )
            st.dataframe(اطول, hide_index=True, width="stretch")
        with عمود_ب:
            st.markdown("**📐 أقصر 10 آيات (بعدد الكلمات)**")
            اقصر = df_لوحة.nsmallest(10, "n_words")[["surah_name", "ref", "n_words"]].rename(
                columns={"surah_name": "السورة", "ref": "الآية", "n_words": "عدد الكلمات"}
            )
            st.dataframe(اقصر, hide_index=True, width="stretch")

        st.markdown("---")

        # --- الكلمات الأكثر تكراراً ---
        st.markdown("**🔤 أعلى 15 كلمة تكراراً**")
        عدد_لوحة = st.slider("عدد الكلمات المعروضة:", 5, 30, 15, key="عدد_لوحة")
        df_تكرار_لوحة = pd.DataFrame(
            Counter(كل_كلمات_لوحة).most_common(عدد_لوحة), columns=["الكلمة", "التكرار"]
        )
        st.bar_chart(df_تكرار_لوحة.set_index("الكلمة"))

        st.markdown("---")

        # --- التوزيع المكي/المدني ---
        عمود_ج, عمود_د = st.columns(2)
        with عمود_ج:
            st.markdown("**🕋 عدد السور: مكية / مدنية**")
            سور_فريدة = df_لوحة.drop_duplicates("surah_id").copy()
            سور_فريدة["النوع"] = سور_فريدة["surah_type"].map(النوع_عربي)
            st.bar_chart(سور_فريدة["النوع"].value_counts())
        with عمود_د:
            st.markdown("**📜 عدد الكلمات: مكية / مدنية**")
            df_لوحة_نوع = df_لوحة.copy()
            df_لوحة_نوع["النوع"] = df_لوحة_نوع["surah_type"].map(النوع_عربي)
            st.bar_chart(df_لوحة_نوع.groupby("النوع")["n_words"].sum())

        st.markdown("---")

        # --- أكثر السور من حيث عدد الكلمات ---
        st.markdown("**📚 أكثر 10 سور من حيث عدد الكلمات**")
        كلمات_حسب_سورة = (
            df_لوحة.groupby("surah_name")["n_words"].sum().sort_values(ascending=False).head(10)
        )
        st.bar_chart(كلمات_حسب_سورة)

        st.markdown("---")

        # --- توزيع أطوال الآيات ---
        st.markdown("**📊 توزيع أطوال الآيات (عدد الكلمات لكل آية)**")
        st.caption("يوضح هذا الرسم عدد الآيات حسب طولها بالكلمات (بعد التطبيع اللغوي).")
        توزيع_اطوال = df_لوحة["n_words"].value_counts().sort_index()
        st.bar_chart(توزيع_اطوال)

# ---------------------------------------------------------------------------
# تبويب 1: عرض النص ضمن النطاق المختار
# ---------------------------------------------------------------------------

with تبويب_نص:
    st.subheader(f"النص القرآني — {len(scope_df)} آية ضمن النطاق المختار")
    for surah_id in سور_مختارة:
        surah_info = next(s for s in corpus if s["id"] == surah_id)
        with st.expander(
            f'سورة {surah_info["name"]} ({surah_info["transliteration"]}) '
            f'— {surah_info["type"]} — {surah_info["total_verses"]} آية',
            expanded=(len(سور_مختارة) == 1),
        ):
            for _, ligne in scope_df[scope_df["surah_id"] == surah_id].iterrows():
                st.markdown(
                    f'<div class="quran-text">﴿{ligne["text"]}﴾ '
                    f'<span style="font-size:14px;color:gray">({ligne["ayah"]})</span></div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# تبويب 2: البحث والتوافق (concordance)
# ---------------------------------------------------------------------------

with تبويب_بحث:
    st.subheader("البحث عن كلمة ضمن النطاق المختار")
    كلمة_بحث = st.text_input("أدخل كلمة للبحث عنها (بالعربية):", value="الله")

    if كلمة_بحث:
        كلمة_منظمة = normaliser(كلمة_بحث)
        نتائج = scope_df[scope_df["text_norm"].apply(
            lambda t: كلمة_منظمة in araby.tokenize(t)
        )]
        st.success(f"تم العثور على {len(نتائج)} آية ضمن النطاق المختار")
        for _, ligne in نتائج.iterrows():
            st.markdown(
                f'**[{ligne["surah_name"]} — {ligne["ref"]}]** '
                f'<span class="quran-text">﴿{ligne["text"]}﴾</span>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# تبويب 3: التكرار المعجمي
# ---------------------------------------------------------------------------

with تبويب_تكرار:
    st.subheader("الكلمات الأكثر تكراراً ضمن النطاق المختار")
    كل_الكلمات = []
    for texte in scope_df["text_norm"]:
        كل_الكلمات.extend(araby.tokenize(texte))
    تكرارات = Counter(كل_الكلمات)

    عدد = st.slider("عدد الكلمات المعروضة:", 5, 50, 15)
    df_تكرار = pd.DataFrame(تكرارات.most_common(عدد), columns=["الكلمة", "عدد التكرارات"])
    st.bar_chart(df_تكرار.set_index("الكلمة"))
    st.dataframe(df_تكرار, width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# تبويب 4: التشابه النصي (TF-IDF + جيب التمام)
# ---------------------------------------------------------------------------

with تبويب_تشابه:
    st.subheader("الآيات الأقرب لغوياً لآية مرجعية")
    st.caption(
        "يقيس هذا المؤشر التقارب المعجمي السطحي (اشتراك الكلمات)، وليس "
        "القرابة الموضوعية أو العقدية بين الآيات."
    )

    عمود1, عمود2 = st.columns(2)
    with عمود1:
        سورة_مرجعية = st.selectbox("السورة:", options_sourates, key="sim_surah")
    with عمود2:
        surah_id_ref = id_depuis_option[سورة_مرجعية]
        max_ayah = next(s["total_verses"] for s in corpus if s["id"] == surah_id_ref)
        آية_مرجعية = st.number_input("رقم الآية:", 1, max_ayah, 1)

    بحث_شامل = st.checkbox("ابحث في كامل القرآن (خارج النطاق المحدد أيضاً)", value=True)

    if st.button("🔍 عرض الآيات الأقرب"):
        idx_ref = index_df[
            (index_df["surah_id"] == surah_id_ref) & (index_df["ayah"] == آية_مرجعية)
        ].index[0]
        similarites = cosine_similarity(matrice_tfidf[idx_ref], matrice_tfidf).flatten()

        resultats = index_df.copy()
        resultats["score"] = similarites
        if not بحث_شامل:
            resultats = resultats[resultats["surah_id"].isin(سور_مختارة)]
        resultats = resultats[resultats.index != idx_ref].sort_values("score", ascending=False).head(10)

        verset_ref = index_df.loc[idx_ref]
        st.info(f'الآية المرجعية: ﴿{verset_ref["text"]}﴾ — [{verset_ref["surah_name"]} {verset_ref["ref"]}]')

        for _, ligne in resultats.iterrows():
            st.markdown(
                f'**[{ligne["surah_name"]} — {ligne["ref"]}]** (تشابه: `{ligne["score"]:.2f}`) '
                f'<span class="quran-text">﴿{ligne["text"]}﴾</span>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# تبويب 5: توليد تقرير PDF
# ---------------------------------------------------------------------------

def ar(texte: str) -> str:
    """يهيئ النص العربي (تشكيل الحروف + اتجاه الكتابة) لعرضه في PDF."""
    return get_display(arabic_reshaper.reshape(escape(str(texte))))


@st.cache_resource
def enregistrer_polices():
    pdfmetrics.registerFont(TTFont("Amiri", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("Amiri-Bold", str(FONT_BOLD)))
    return True


def generer_pdf(sourates_selectionnees, mot_recherche, resultats_recherche,
                 df_frequences, verset_ref_info, resultats_similarite) -> bytes:
    enregistrer_polices()

    style_titre = ParagraphStyle("titre", fontName="Amiri-Bold", fontSize=18,
                                  alignment=TA_CENTER, leading=26, spaceAfter=14)
    style_section = ParagraphStyle("section", fontName="Amiri-Bold", fontSize=14,
                                    alignment=TA_RIGHT, leading=22, spaceBefore=14, spaceAfter=8)
    style_normal = ParagraphStyle("normal", fontName="Amiri", fontSize=12,
                                   alignment=TA_RIGHT, leading=20)
    style_verset = ParagraphStyle("verset", fontName="Amiri", fontSize=13,
                                   alignment=TA_RIGHT, leading=24, textColor="#1a1a1a")
    style_avertissement = ParagraphStyle("avert", fontName="Amiri", fontSize=10,
                                          alignment=TA_RIGHT, leading=16, textColor="#7a4b00")

    story = []
    story.append(Paragraph(ar("تقرير تحليل لغوي للنص القرآني"), style_titre))
    story.append(Paragraph(ar(
        "تنبيه منهجي: هذا تقرير وصفي إحصائي (تكرار معجمي وتشابه نصي سطحي) "
        "لا يمثل بأي حال تفسيراً دينياً، ويجب أن يخضع أي استخدام تأويلي "
        "لنتائجه لمراجعة أهل الاختصاص الشرعي واللغوي."
    ), style_avertissement))
    story.append(Spacer(1, 0.5 * cm))

    # نطاق التحليل
    story.append(Paragraph(ar("نطاق التحليل"), style_section))
    noms_sourates = ", ".join(
        f'{s["name"]} ({s["id"]})' for s in corpus if s["id"] in sourates_selectionnees
    )
    story.append(Paragraph(ar(f"السور المشمولة: {noms_sourates}"), style_normal))
    story.append(Paragraph(ar(f"عدد السور: {len(sourates_selectionnees)}"), style_normal))

    # التكرار المعجمي
    story.append(Paragraph(ar("الكلمات الأكثر تكراراً"), style_section))
    donnees_tableau = [[ar("عدد التكرارات"), ar("الكلمة")]]
    for _, row in df_frequences.iterrows():
        donnees_tableau.append([str(row["عدد التكرارات"]), ar(row["الكلمة"])])
    tableau = Table(donnees_tableau, colWidths=[6 * cm, 6 * cm])
    tableau.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Amiri"),
        ("FONTNAME", (0, 0), (-1, 0), "Amiri-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, "#999999"),
        ("BACKGROUND", (0, 0), (-1, 0), "#e8e8e8"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(tableau)

    # البحث والتوافق
    if mot_recherche:
        story.append(Paragraph(ar(f"نتائج البحث عن كلمة: {mot_recherche}"), style_section))
        if len(resultats_recherche) == 0:
            story.append(Paragraph(ar("لم يتم العثور على نتائج."), style_normal))
        for _, ligne in resultats_recherche.head(15).iterrows():
            story.append(Paragraph(
                ar(f'[{ligne["surah_name"]} {ligne["ref"]}] ﴿{ligne["text"]}﴾'), style_verset
            ))

    # التشابه النصي
    if verset_ref_info is not None:
        story.append(PageBreak())
        story.append(Paragraph(ar("التشابه النصي"), style_section))
        story.append(Paragraph(
            ar(f'الآية المرجعية [{verset_ref_info["surah_name"]} {verset_ref_info["ref"]}]: '
               f'﴿{verset_ref_info["text"]}﴾'),
            style_verset,
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(ar("أقرب الآيات لغوياً:"), style_normal))
        for _, ligne in resultats_similarite.iterrows():
            story.append(Paragraph(
                ar(f'[{ligne["surah_name"]} {ligne["ref"]}] (تشابه: {ligne["score"]:.2f}) '
                   f'﴿{ligne["text"]}﴾'),
                style_verset,
            ))

    from io import BytesIO
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    doc.build(story)
    return buffer.getvalue()


with تبويب_تقرير:
    st.subheader("توليد تقرير PDF شامل")
    st.caption("يجمع هذا التقرير التكرار المعجمي، نتائج البحث، والتشابه النصي ضمن النطاق المختار.")

    mot_rapport = st.text_input("كلمة للبحث عنها في التقرير:", value="الرحمن", key="mot_rapport")
    عمود1, عمود2 = st.columns(2)
    with عمود1:
        sourate_rapport = st.selectbox("سورة الآية المرجعية للتشابه:", options_sourates, key="rapport_surah")
    with عمود2:
        sid_rapport = id_depuis_option[sourate_rapport]
        max_ayah_rapport = next(s["total_verses"] for s in corpus if s["id"] == sid_rapport)
        ayah_rapport = st.number_input("رقم الآية:", 1, max_ayah_rapport, 1, key="rapport_ayah")

    if st.button("📄 توليد التقرير"):
        with st.spinner("جارٍ توليد ملف PDF..."):
            كل_الكلمات = []
            for texte in scope_df["text_norm"]:
                كل_الكلمات.extend(araby.tokenize(texte))
            df_freq_rapport = pd.DataFrame(
                Counter(كل_الكلمات).most_common(15), columns=["الكلمة", "عدد التكرارات"]
            )

            mot_norm_rapport = normaliser(mot_rapport) if mot_rapport else ""
            resultats_recherche_rapport = scope_df[
                scope_df["text_norm"].apply(lambda t: mot_norm_rapport in araby.tokenize(t))
            ] if mot_rapport else pd.DataFrame()

            idx_ref_rapport = index_df[
                (index_df["surah_id"] == sid_rapport) & (index_df["ayah"] == ayah_rapport)
            ].index[0]
            sims_rapport = cosine_similarity(matrice_tfidf[idx_ref_rapport], matrice_tfidf).flatten()
            res_sim_rapport = index_df.copy()
            res_sim_rapport["score"] = sims_rapport
            res_sim_rapport = res_sim_rapport[res_sim_rapport.index != idx_ref_rapport]
            res_sim_rapport = res_sim_rapport.sort_values("score", ascending=False).head(5)
            verset_ref_rapport = index_df.loc[idx_ref_rapport]

            pdf_bytes = generer_pdf(
                سور_مختارة, mot_rapport, resultats_recherche_rapport,
                df_freq_rapport, verset_ref_rapport, res_sim_rapport,
            )

        st.success("تم توليد التقرير بنجاح.")
        st.download_button(
            "⬇️ تحميل التقرير (PDF)", data=pdf_bytes,
            file_name="تقرير_تحليل_القرآن.pdf", mime="application/pdf",
        )

st.markdown("---")
st.caption(
    "أداة وصفية إحصائية لأغراض بحثية وتقنية فقط. لا تُغني عن التفسير "
    "المعتمد على علم شرعي أصيل. يُرجى مراجعة أهل الاختصاص لأي استخدام تأويلي."
)
