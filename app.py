import os
import json
import requests
import streamlit as st
from PIL import Image
import google.generativeai as genai
from pydantic import BaseModel, Field

# --- CREDENTIALS & CONSTANTS ---
BOOKSRUN_API_KEY = "j69yick1gexf8blbdtix"
BOOKSRUN_AFK = "31443"

st.set_page_config(
    page_title="Book Scout",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Fetch Gemini API Key from Streamlit Secrets or manual fallback
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
else:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

min_profit_threshold = st.sidebar.slider("Min Target Payout ($)", min_value=0.50, max_value=20.00, value=3.00, step=0.50)

# --- STRUCTURED SCHEMA ---
class DetectedBook(BaseModel):
    title: str = Field(description="Exact book title visible on the cover or spine.")
    author: str = Field(description="Author's name if visible, or empty string.")
    binding: str = Field(description="Format: 'Hardcover', 'Paperback', or 'Unknown'")

class BookListExtraction(BaseModel):
    books: list[DetectedBook]

# --- PIPELINE ENGINES ---
def extract_books_with_vision(image: Image.Image, api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "Identify every individual, clearly readable book in this photo. "
        "Extract the main title, author name (if discernible), and whether it appears "
        "to be a Hardcover or Paperback based on physical binding and edges. "
        "Ignore board games, notebooks, toys, or illegible items."
    )
    response = model.generate_content(
        [prompt, image],
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": BookListExtraction,
            "temperature": 0.1
        }
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

# --- INTERFACE ---
st.title("📚 Scout & Flip")
st.caption("Visual Garage Sale Scanner powered by BooksRun")

if not gemini_key:
    st.info("⚠️ Please enter your Gemini API key in the left sidebar or configure Streamlit Secrets.")
    st.stop()

tab_camera, tab_manual = st.tabs(["📸 Snap Photo", "⌨️ Manual ISBNs"])

with tab_camera:
    camera_photo = st.camera_input("Capture book covers or shelf:")
    file_photo = st.file_uploader("Or upload from gallery:", type=["jpg", "jpeg", "png"])
    active_image = camera_photo or file_photo

    if active_image and st.button("🚀 Analyze & Check Offers", use_container_width=True, type="primary"):
        img = Image.open(active_image)
        with st.spinner("AI scanning covers and extracting titles..."):
            detected_books = extract_books_with_vision(img, gemini_key)
        
        if not detected_books:
            st.warning("No readable books found. Try a closer angle.")
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
