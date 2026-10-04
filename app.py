import os
import json
import requests
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- APP VERSION & CONFIGURATION ---
APP_VERSION = "v3.1"
BOOKSRUN_API_KEY = "j69yick1gexf8blbdtix"
BOOKSRUN_AFK = "31443"

st.set_page_config(
    page_title=f"Book Scout {APP_VERSION}",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Fetch Gemini API Key from Streamlit Secrets or sidebar
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
else:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

min_profit_threshold = st.sidebar.slider(
    "Min Target Payout ($)", 
    min_value=0.50, 
    max_value=20.00, 
    value=3.00, 
    step=0.50
)

# --- IMAGE COMPRESSION HELPER ---
def optimize_image_for_api(image: Image.Image, max_dim: int = 1600) -> Image.Image:
    """Downsamples massive smartphone photos to speed up transmission and prevent timeouts."""
    img = image.convert("RGB")
    width, height = img.size
    
    if max(width, height) > max_dim:
        scale = max_dim / float(max(width, height))
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img

# --- STRUCTURED SCHEMA ---
class DetectedBook(BaseModel):
    title: str = Field(description="Exact book title visible on the cover or spine.")
    author: str = Field(description="Author's name if visible, or empty string.")
    binding: str = Field(description="Format: 'Hardcover', 'Paperback', or 'Unknown'")

class BookListExtraction(BaseModel):
    books: list[DetectedBook]

# --- PIPELINE ENGINES ---
def extract_books_with_vision(image: Image.Image, api_key: str):
    client = genai.Client(api_key=api_key)
    prompt = (
        "Identify every individual, clearly readable book in this photo. "
        "Extract the main title, author name (if discernible), and whether it appears "
        "to be a Hardcover or Paperback based on physical binding and edges. "
        "Ignore board games, notebooks, toys, or illegible items."
    )
    
    # Process with optimized image
    optimized_img = optimize_image_for_api(image)
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[prompt, optimized_img],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BookListExtraction,
            temperature=0.1
        )
    )
    return json.loads(response.text).get("books", [])

def resolve_isbn(title: str, author: str = "") -> dict:
    query = f"intitle:{title}"
    if author:
        query += f" inauthor:{author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={requests.utils.quote(query)}&maxResults=3&printType=books"
    try:
        res = requests.get(url, timeout=5).json()
        for item in res.get("items", []):
            info = item.get("volumeInfo", {})
            for ident in info.get("industryIdentifiers", []):
                if ident.get("type") in ["ISBN_13", "ISBN_10"]:
                    return {"isbn": ident.get("identifier"), "title": info.get("title", title)}
    except Exception:
        pass
    return {}

def check_booksrun_quote(isbn: str) -> dict:
    url = f"https://booksrun.com/api/price/sell/{isbn}?key={BOOKSRUN_API_KEY}"
    try:
        res = requests.get(url, timeout=6).json()
        if res.get("result", {}).get("status") == "success":
            return res["result"].get("text", {})
    except Exception:
        pass
    return {}

# --- HEADER & VERSION TAG ---
col_head, col_ver = st.columns([4, 1])
with col_head:
    st.title("📚 Scout & Flip")
    st.caption("Visual Garage Sale Scanner powered by BooksRun")
with col_ver:
    st.markdown(f"<span style='float:right; background:#2e303d; color:#00e676; padding:4px 8px; border-radius:6px; font-weight:bold;'>{APP_VERSION}</span>", unsafe_allow_html=True)

if not gemini_key:
    st.info("⚠️ Please enter your Gemini API key in the left sidebar or configure Streamlit Secrets.")
    st.stop()

tab_camera, tab_manual = st.tabs(["📸 Snap / Upload", "⌨️ Manual ISBNs"])

with tab_camera:
    st.markdown("**Take Photo (Uses Rear Camera & Lens Zoom):**")
    file_photo = st.file_uploader("Tap to open phone camera or select photo:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    with st.expander("Or use embedded live browser viewfinder"):
        camera_photo = st.camera_input("Browser viewfinder", label_visibility="collapsed")

    active_image = file_photo or camera_photo

    if active_image and st.button("🚀 Analyze & Check Offers", use_container_width=True, type="primary"):
        img = Image.open(active_image)
        with st.spinner("AI scanning covers and extracting titles with Gemini 3.8 Flash..."):
            try:
                detected_books = extract_books_with_vision(img, gemini_key)
            except Exception as e:
                st.error(f"Vision analysis error: {e}")
                detected_books = []
        
        if not detected_books:
            st.warning("No readable books found. Try taking a closer photo.")
        else:
            st.subheader(f"Found {len(detected_books)} Potential Books")
            total_cart = 0.0

            for book in detected_books:
                t = book.get("title")
                a = book.get("author", "")
                fmt = book.get("binding", "Unknown")
                
                with st.spinner(f"Checking quote: {t}..."):
                    resolved = resolve_isbn(t, a)
                    isbn = resolved.get("isbn")
                    if not isbn:
                        st.markdown(f"❌ **{t}** — *Could not resolve ISBN*")
                        continue
                    
                    quotes = check_booksrun_quote(isbn)
                    good_offer = float(quotes.get("Good", 0.0))
                    
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        c1.markdown(f"**{t}**\n\n`{isbn}` ({fmt})")
                        if good_offer >= min_profit_threshold:
                            c2.success(f"**${good_offer:.2f}**")
                            cart_url = f"https://booksrun.com/api/cart/sell/add/{isbn}:good?afk={BOOKSRUN_AFK}"
                            c2.link_button("🛒 Add", cart_url, use_container_width=True)
                            total_cart += good_offer
                        elif good_offer > 0:
                            c2.warning(f"${good_offer:.2f}")
                        else:
                            c2.markdown("<span style='color: gray;'>$0.00</span>", unsafe_allow_html=True)

            st.divider()
            st.metric("Viable Payout Total", f"${total_cart:.2f}")
            if total_cart >= 15.00:
                st.success("🎉 Carton Threshold Met ($15+)! Free shipping label qualified.")
            else:
                st.info(f"Cart total is ${total_cart:.2f}. Needs **${15.00 - total_cart:.2f} more** for free shipping.")

with tab_manual:
    raw_isbns = st.text_area("Paste ISBNs:", "9780135188743, 9780744059793")
    if st.button("Check Manual ISBNs", use_container_width=True):
        clean_list = [i.strip().replace("-", "") for i in raw_isbns.replace("\n", ",").split(",") if i.strip()]
        for isbn in clean_list:
            quotes = check_booksrun_quote(isbn)
            good_offer = float(quotes.get("Good", 0.0))
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.write(f"ISBN: `{isbn}`")
                if good_offer > 0:
                    c2.success(f"${good_offer:.2f}")
                    cart_url = f"https://booksrun.com/api/cart/sell/add/{isbn}:good?afk={BOOKSRUN_AFK}"
                    c2.link_button("Add", cart_url)
                else:
                    c2.write("$0.00")
