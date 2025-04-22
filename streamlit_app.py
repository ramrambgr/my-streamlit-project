import streamlit as st
from PIL import Image, ExifTags
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ctransformers import AutoModelForCausalLM
import PyPDF2
import docx

st.set_page_config(page_title="AI Assistant Jurnalis", layout="wide")

# --- Load BLIP Model for Captioning ---
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def caption_image(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)

# --- Ekstraksi Metadata Gambar (EXIF) ---
def extract_exif_data(image):
    exif_data = {}
    try:
        raw_exif = image._getexif()
        if raw_exif:
            exif = {
                ExifTags.TAGS.get(tag, tag): value
                for tag, value in raw_exif.items()
                if tag in ExifTags.TAGS
            }
            exif_data["Tanggal"] = exif.get("DateTimeOriginal", "Tidak tersedia")
            exif_data["Kamera"] = exif.get("Model", "Tidak tersedia")
            gps_info = exif.get("GPSInfo")
            if gps_info:
                gps_data = {}
                for key in gps_info.keys():
                    decoded_key = ExifTags.GPSTAGS.get(key, key)
                    gps_data[decoded_key] = gps_info[key]
                exif_data["GPS"] = gps_data
            else:
                exif_data["GPS"] = "Tidak tersedia"
        else:
            exif_data["Info"] = "Tidak ada metadata EXIF."
    except Exception as e:
        exif_data["Error"] = str(e)
    return exif_data

# --- Load LLaMA Model via ctransformers ---
@st.cache_resource
def load_llama_model(model_file="llama-2-7b-chat.ggmlv3.q3_K_L.bin", max_tokens=1500):
    return AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-Chat-GGML",
        model_file=model_file,
        model_type="llama",
        max_new_tokens=max_tokens,
        temperature=0.5,
        context_length=2048
    )

# --- Agent Planner ---
def agent_planner(caption: str, user_input: str) -> dict:
    steps = []
    questions = []
    goals = []

    if "video" in user_input.lower():
        goals.append("Tulis skrip narasi untuk video")
    elif "investigasi" in user_input.lower():
        goals.append("Tulis artikel investigatif dengan alur kronologis")
        questions.append("Apakah Anda memiliki data waktu atau lokasi kejadian?")
    else:
        goals.append("Tulis artikel berita yang lengkap, mendalam, dan faktual")

    if "identitas" in user_input.lower() or "siapa" in user_input.lower():
        steps.append("Analisa gambar untuk mendeteksi tokoh atau objek penting")
    if not caption or caption.strip() == "":
        steps.append("Buat deskripsi gambar lebih dulu")

    return {
        "goals": goals,
        "steps": steps,
        "questions": questions
    }

# --- Generate Prompt (diperbaiki dengan pemotongan referensi dan larangan menyalin) ---
def generate_prompt(caption, user_input, language="id", reference_text=None):
    MAX_REF_CHARS = 2000
    reference_section = ""

    if reference_text:
        if len(reference_text) > MAX_REF_CHARS:
            reference_text = reference_text[:MAX_REF_CHARS] + "\n... [Referensi dipotong agar tidak terlalu panjang]"

        reference_section = f"""
Berikut adalah referensi tulisan dari pengguna. Referensi ini **hanya digunakan untuk memahami gaya penulisan** dan **bukan untuk disalin langsung atau dianggap sebagai isi berita**:
\"\"\"
{reference_text}
\"\"\"
"""

    if language == "id":
        return f"""
Kamu adalah seorang jurnalis profesional yang mahir dalam menulis berita berkualitas tinggi dalam Bahasa Indonesia.

Berikut adalah deskripsi visual dari gambar yang disediakan:

"{caption}"

{reference_section}

Dan berikut permintaan dari pengguna:
"{user_input}"

Tugasmu:
- Tulis artikel berita panjang berdasarkan permintaan pengguna.
- Gunakan referensi hanya sebagai inspirasi gaya penulisan, bukan sebagai sumber fakta.
- Jangan menyalin isi referensi secara langsung.
- Fokus pada kronologi, penyebab, dampak sosial, dan respons pemerintah sesuai instruksi pengguna.

Buatlah artikel berita panjang yang informatif dan mudah dipahami, dengan struktur sebagai berikut:

---

**🕴️ Judul:**
Tulis judul yang singkat, jelas, dan menarik.

**📌 Pembuka (Lead):**
Tuliskan paragraf pembuka yang menjawab 5W+1H secara padat.

**📖 Isi Berita (minimal 1000 kata):**
Rinci kronologi, penyebab, dampak, kutipan, dan respons terkait. Gunakan kalimat aktif, bahasa lugas, dan struktur jurnalistik.

**🗾 Penutup:**
Simpulkan peristiwa secara ringkas, sertakan ajakan atau pesan jika relevan.

---

Gunakan gaya bahasa jurnalistik Indonesia yang alami dan hindari kalimat terjemahan atau yang terlalu puitis.
"""
    else:
        return f"... (English version if needed)"

# --- Generate Response ---
def generate_response(model, prompt, system_prompt="Kamu adalah AI Assistant yang membantu dalam jurnalisme dan penulisan berita."):
    formatted_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>\n\n{prompt} [/INST]"""
    return model(formatted_prompt)

# --- UI Start ---
st.title("📰 AI Assistant Jurnalis")
st.write("Mengubah gambar & ide menjadi artikel berita panjang. Didukung oleh BLIP + LLaMA + EXIF + Referensi.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

try:
    llama_model = load_llama_model()
    st.sidebar.success("✅ LLaMA siap digunakan.")
except Exception as e:
    st.sidebar.error(f"Gagal load LLaMA: {e}")
    st.stop()

processor, caption_model = load_caption_model()

# --- File uploader without max_size argument ---
uploaded_file = st.file_uploader("📷 Upload gambar kejadian atau ilustrasi:", type=["jpg", "jpeg", "png"])

image_caption = None

if uploaded_file:
    file_size = len(uploaded_file.getvalue())  # Dapatkan ukuran file dalam byte
    max_size = 10 * 1024 * 1024  # 10 MB
    
    if file_size > max_size:
        st.warning("⚠️ File terlalu besar! Maksimal ukuran file adalah 10 MB.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        exif = extract_exif_data(image)

        with st.expander("📸 Metadata Gambar (EXIF)", expanded=False):
            for k, v in exif.items():
                if isinstance(v, dict):
                    st.markdown(f"**{k}:**")
                    for subk, subv in v.items():
                        st.markdown(f"- {subk}: {subv}")
                else:
                    st.markdown(f"**{k}:** {v}")

        with st.spinner("🔍 Menganalisis gambar..."):
            image_caption = caption_image(image, processor, caption_model)
            if exif:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in exif.items() if not isinstance(v, dict)])
                image_caption += f"\n\n[Metadata Gambar]\n{metadata_text}"

            st.success("✅ Gambar dianalisis.")
            st.markdown(f"**Caption Gambar:** *{image_caption}*")

language = st.selectbox("🌐 Pilih Bahasa Output Artikel", ["Bahasa Indonesia", "English"])
lang_code = "id" if language == "Bahasa Indonesia" else "en"

user_input = st.text_area("🗣️ Apa yang ingin kamu tulis atau tanyakan?", height=100)

ref_file = st.file_uploader("📄 Upload artikel referensi (opsional):", type=["txt", "docx", "pdf"])
use_reference = False
ref_text = ""

if ref_file is not None:
    try:
        if ref_file.type == "text/plain":
            ref_text = ref_file.read().decode("utf-8")
        elif ref_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(ref_file)
            ref_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ref_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(ref_file)
            ref_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            st.warning("⚠️ Format file belum didukung.")

        if ref_text.strip():
            st.success("✅ Referensi berhasil dibaca.")
            use_reference = st.checkbox("📌 Gunakan referensi ini sebagai gaya penulisan", value=True)

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

if st.button("📨 Buat Artikel Panjang"):
    if not user_input.strip():
        st.warning("Tolong masukkan perintah atau pertanyaan.")
    else:
        with st.spinner("✍️ Menulis artikel 1000 kata..."):
            caption_text = image_caption if image_caption else "Tidak ada gambar."
            prompt = generate_prompt(
                caption_text,
                user_input,
                language=lang_code,
                reference_text=ref_text if use_reference else None
            )

            agent_action = agent_planner(caption_text, user_input)

            if agent_action["questions"]:
                for q in agent_action["questions"]:
                    st.info(f"🤖 Pertanyaan tambahan: {q}")

            if agent_action["steps"]:
                for s in agent_action["steps"]:
                    st.markdown(f"🧭 Langkah yang akan dilakukan: {s}")

            if agent_action["goals"]:
                st.markdown("🎯 Tujuan AI:")
                for g in agent_action["goals"]:
                    st.markdown(f"- {g}")

            result = generate_response(llama_model, prompt)
            st.session_state.chat_history.append({
                "user": user_input,
                "caption": caption_text,
                "result": result,
                "language": language
            })
            st.success("✅ Artikel panjang selesai dibuat!")

if st.session_state.chat_history:
    st.subheader("📜 Riwayat Artikel")
    for idx, chat in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"### 📝 Artikel #{len(st.session_state.chat_history)-idx+1}")
        st.markdown(f"**🧑 Permintaan:** {chat['user']}")
        if 'caption' in chat:
            st.markdown(f"**🖼️ Caption Gambar:** *{chat['caption']}*")
        st.markdown(f"**🌐 Bahasa:** {chat['language']}")
        st.markdown(f"**🤖 Hasil Artikel:**\n\n{chat['result']}")
        st.markdown("---")

    full_text = "\n\n".join([f"🧑 {c['user']}\n🌐 Bahasa: {c['language']}\n🖼️ {c['caption']}\n🤖 {c['result']}" for c in st.session_state.chat_history])
    st.download_button("💾 Download Semua Artikel", data=full_text.encode("utf-8"), file_name="arsip_berita_ai.txt", mime="text/plain")

st.sidebar.title("💡 Contoh Prompt")
st.sidebar.markdown("""
- Tulis berita panjang tentang kebakaran berdasarkan gambar.
- Buat artikel investigatif mendalam.
- Tulis artikel opini berbasis gambar.
- Tulis berita viral 1000 kata.
""")
