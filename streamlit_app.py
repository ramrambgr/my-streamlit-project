import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="AI Assistant Jurnalis", layout="wide")

# --- Load BLIP Model for Captioning (Using Hugging Face API) ---
@st.cache_resource
def load_caption_model():
    return pipeline("image-captioning", model="Salesforce/blip-image-captioning-base")

def caption_image(image_path, model):
    caption = model(image_path)
    return caption[0]["caption"]

# --- Agent Planner ---
def agent_planner(caption: str, user_input: str) -> dict:
    steps = []
    goals = []

    if "video" in user_input.lower():
        goals.append("Tulis skrip narasi untuk video")
    elif "investigasi" in user_input.lower():
        goals.append("Tulis artikel investigatif dengan alur kronologis")
    else:
        goals.append("Tulis artikel berita yang lengkap, mendalam, dan faktual")

    if "identitas" in user_input.lower() or "siapa" in user_input.lower():
        steps.append("Analisa gambar untuk mendeteksi tokoh atau objek penting")
    if not caption or caption.strip() == "":
        steps.append("Buat deskripsi gambar lebih dulu")

    return {
        "goals": goals,
        "steps": steps
    }

# --- Generate Prompt ---
def generate_prompt(caption, user_input, language="id"):
    return f"""
Kamu adalah seorang jurnalis profesional yang mahir dalam menulis berita berkualitas tinggi dalam Bahasa Indonesia.

Berikut adalah deskripsi visual dari gambar yang disediakan:
"{caption}"

Dan berikut permintaan dari pengguna:
"{user_input}"

Tugasmu:
- Tulis artikel berita panjang berdasarkan permintaan pengguna.
- Gunakan gaya bahasa jurnalistik Indonesia yang alami.

---

**🕴️ Judul:**
Tulis judul yang singkat, jelas, dan menarik.

**📌 Pembuka (Lead):**
Tuliskan paragraf pembuka yang menjawab 5W+1H secara padat.

**📖 Isi Berita (minimal 1000 kata):**
Rinci kronologi, penyebab, dampak, kutipan, dan respons terkait.

**🗾 Penutup:**
Simpulkan peristiwa secara ringkas.

"""

# --- Generate Response ---
def generate_response(model, prompt, system_prompt="Kamu adalah AI Assistant yang membantu dalam jurnalisme dan penulisan berita."):

    formatted_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>\n\n{prompt} [/INST]"""
    return model(formatted_prompt)

# --- UI Start ---
st.title("📰 AI Assistant Jurnalis")
st.write("Mengubah gambar menjadi artikel berita panjang.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

try:
    caption_model = load_caption_model()
    st.sidebar.success("✅ Model untuk captioning siap digunakan.")
except Exception as e:
    st.sidebar.error(f"Gagal load model captioning: {e}")
    st.stop()

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

        with st.spinner("🔍 Menganalisis gambar..."):
            image_caption = caption_image(image, caption_model)

            st.success("✅ Gambar dianalisis.")
            st.markdown(f"**Caption Gambar:** *{image_caption}*")

user_input = st.text_area("🗣️ Apa yang ingin kamu tulis atau tanyakan?", height=100)

if st.button("📨 Buat Artikel Panjang"):
    if not user_input.strip():
        st.warning("Tolong masukkan perintah atau pertanyaan.")
    else:
        with st.spinner("✍️ Menulis artikel 1000 kata..."):
            caption_text = image_caption if image_caption else "Tidak ada gambar."
            prompt = generate_prompt(caption_text, user_input)

            agent_action = agent_planner(caption_text, user_input)

            if agent_action["steps"]:
                for s in agent_action["steps"]:
                    st.markdown(f"🧭 Langkah yang akan dilakukan: {s}")

            if agent_action["goals"]:
                st.markdown("🎯 Tujuan AI:")
                for g in agent_action["goals"]:
                    st.markdown(f"- {g}")

            result = generate_response(caption_model, prompt)
            st.success("✅ Artikel panjang selesai dibuat!")
            st.write(result)
