import streamlit as st
import pandas as pd
from io import BytesIO
import pypdf
import docx
import openpyxl
import chardet
from typing import List, Dict, Any
import json

# Sayfa yapılandırması
st.set_page_config(
    page_title="Doküman Chunking Uygulaması",
    page_icon="📄",
    layout="wide"
)

# CSS ile stil ekleme
st.markdown("""
    <style>
    .chunk-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
    }
    .chunk-header {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .chunk-text {
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .stats-container {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .file-info {
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #007bff;
    }
    </style>
""", unsafe_allow_html=True)

# Başlık ve açıklama
st.title("📄 Gelişmiş Doküman Chunking Uygulaması")
st.markdown("### Metin belgelerinizi anlamlı parçalara bölün")

# Session state için başlangıç değerleri
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []


def detect_encoding(file_bytes):
    """Dosya encoding'ini tespit et"""
    result = chardet.detect(file_bytes)
    return result['encoding'] if result['encoding'] else 'utf-8'


def read_pdf(file):
    """PDF dosyasını oku"""
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text, None, None


def read_docx(file):
    """DOCX dosyasını oku"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text, None, None


def read_txt(file):
    """TXT dosyasını oku"""
    file_bytes = file.read()
    encoding = detect_encoding(file_bytes)
    try:
        text = file_bytes.decode(encoding)
    except:
        text = file_bytes.decode('utf-8', errors='ignore')
    return text, None, None


def read_csv(file):
    """CSV dosyasını oku ve sütun bilgilerini sakla"""
    try:
        df = pd.read_csv(file)
        text = df.to_string()
        columns = df.columns.tolist()
        return text, df, columns
    except Exception as e:
        st.error(f"CSV okuma hatası: {str(e)}")
        return "", None, None


def read_excel(file):
    """Excel dosyasını oku ve sütun bilgilerini sakla"""
    try:
        df = pd.read_excel(file)
        text = df.to_string()
        columns = df.columns.tolist()
        return text, df, columns
    except Exception as e:
        st.error(f"Excel okuma hatası: {str(e)}")
        return "", None, None


def create_chunks_with_columns(df, columns, chunk_method, chunk_size=500, chunk_overlap=50, separator="\n\n",
                               include_columns_in_text=False):
    """DataFrame'den sütun bilgileriyle chunk'lar oluştur"""
    chunks = []

    if chunk_method == "Satır Bazlı":
        # Her satırı ayrı chunk olarak al
        for idx, row in df.iterrows():
            chunk_data = {}

            # Sütun verilerini ekle
            for col in columns:
                chunk_data[col] = str(row[col]) if pd.notna(row[col]) else ""

            # Text alanını oluştur
            if include_columns_in_text:
                chunk_text = ""
                for col in columns:
                    chunk_text += f"{col}: {row[col]}\n"
                chunk_data['text'] = chunk_text.strip()
            else:
                # Sadece değerleri birleştir
                chunk_data['text'] = " ".join([str(row[col]) for col in columns if pd.notna(row[col])])

            chunks.append(chunk_data)

    elif chunk_method == "Sabit Boyut":
        # Birden fazla satırı birleştirerek chunk oluştur
        current_rows = []
        current_text = ""

        for idx, row in df.iterrows():
            row_text = ""
            if include_columns_in_text:
                for col in columns:
                    row_text += f"{col}: {row[col]}\n"
                row_text += "-" * 30 + "\n"
            else:
                row_text = " ".join([str(row[col]) for col in columns if pd.notna(row[col])]) + "\n"

            if len(current_text) + len(row_text) > chunk_size:
                if current_rows:
                    chunk_data = {}
                    # İlk satırın sütun değerlerini temsili olarak ekle
                    for col in columns:
                        values = [str(r[col]) for r in current_rows if pd.notna(r[col])]
                        chunk_data[col] = values[0] if values else ""
                    chunk_data['text'] = current_text.strip()
                    chunks.append(chunk_data)

                current_rows = [row.to_dict()]
                current_text = row_text
            else:
                current_rows.append(row.to_dict())
                current_text += row_text

        # Kalan veriyi ekle
        if current_rows:
            chunk_data = {}
            for col in columns:
                values = [str(r[col]) for r in current_rows if pd.notna(r.get(col))]
                chunk_data[col] = values[0] if values else ""
            chunk_data['text'] = current_text.strip()
            chunks.append(chunk_data)

    else:  # Ayırıcı Bazlı
        # Grup halinde chunk'lama
        rows_per_chunk = max(1, chunk_size // 100)
        for i in range(0, len(df), rows_per_chunk):
            chunk_rows = df.iloc[i:i + rows_per_chunk]
            chunk_data = {}

            # İlk satırın sütun değerlerini temsili olarak ekle
            first_row = chunk_rows.iloc[0]
            for col in columns:
                chunk_data[col] = str(first_row[col]) if pd.notna(first_row[col]) else ""

            # Text oluştur
            chunk_text = ""
            for idx, row in chunk_rows.iterrows():
                if include_columns_in_text:
                    for col in columns:
                        chunk_text += f"{col}: {row[col]}\n"
                    chunk_text += "-" * 30 + "\n"
                else:
                    chunk_text += " ".join([str(row[col]) for col in columns if pd.notna(row[col])]) + "\n"

            chunk_data['text'] = chunk_text.strip()
            chunks.append(chunk_data)

    return chunks


def create_chunks(text: str, method: str, chunk_size: int = 500,
                  chunk_overlap: int = 50, separator: str = "\n\n") -> List[Dict]:
    """Metni chunk'lara böl"""
    chunks = []

    if method == "Sabit Boyut":
        size = max(1, chunk_size)
        effective_overlap = min(max(0, chunk_overlap), size - 1)
        step = size - effective_overlap

        start = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end]
            chunks.append({'text': chunk_text})
            start += step

    elif method == "Ayırıcı Bazlı":
        size = max(1, chunk_size)
        parts = text.split(separator)
        current_chunk = ""

        for part in parts:
            candidate = (current_chunk + part + separator) if current_chunk else (part + separator)
            if len(candidate) <= size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append({'text': current_chunk.strip()})
                    current_chunk = part + separator
                    if len(current_chunk) > size:
                        # Çok uzun parça için kırpma (nadiren gerekir)
                        while len(current_chunk) > size:
                            chunks.append({'text': current_chunk[:size].strip()})
                            current_chunk = current_chunk[size:]
                else:
                    # İlk parça tek başına limiti aşıyorsa
                    piece = part + separator
                    while len(piece) > size:
                        chunks.append({'text': piece[:size].strip()})
                        piece = piece[size:]
                    current_chunk = piece

        if current_chunk:
            chunks.append({'text': current_chunk.strip()})

        # Overlap uygula (karakter bazlı), boyutu koru
        if chunk_overlap > 0 and len(chunks) > 1:
            effective_overlap = min(max(0, chunk_overlap), size - 1)
            for i in range(1, len(chunks)):
                prev_text = chunks[i - 1]['text']
                prefix = prev_text[-effective_overlap:] if len(prev_text) >= effective_overlap else prev_text
                new_text = (prefix + chunks[i]['text'])
                chunks[i]['text'] = new_text[:size]

    elif method == "Cümle Bazlı":
        size = max(1, chunk_size)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = (current_chunk + sentence + ". ") if current_chunk else (sentence + ". ")
            if len(candidate) <= size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append({'text': current_chunk.strip()})
                    current_chunk = sentence + ". "
                    if len(current_chunk) > size:
                        # Çok uzun tek cümle için kırpma
                        while len(current_chunk) > size:
                            chunks.append({'text': current_chunk[:size].strip()})
                            current_chunk = current_chunk[size:]
                else:
                    piece = sentence + ". "
                    while len(piece) > size:
                        chunks.append({'text': piece[:size].strip()})
                        piece = piece[size:]
                    current_chunk = piece

        if current_chunk:
            chunks.append({'text': current_chunk.strip()})

        # Overlap uygula (karakter bazlı), boyutu koru
        if chunk_overlap > 0 and len(chunks) > 1:
            effective_overlap = min(max(0, chunk_overlap), size - 1)
            for i in range(1, len(chunks)):
                prev_text = chunks[i - 1]['text']
                prefix = prev_text[-effective_overlap:] if len(prev_text) >= effective_overlap else prev_text
                new_text = (prefix + chunks[i]['text'])
                chunks[i]['text'] = new_text[:size]

    return chunks


def process_file(file, chunk_method, chunk_size, chunk_overlap, separator, include_columns_in_text):
    """Tek bir dosyayı işle"""
    file_extension = file.name.split('.')[-1].lower()
    file_name = file.name

    # Dosya türüne göre okuma
    if file_extension == 'pdf':
        text, df, columns = read_pdf(file)
    elif file_extension == 'docx':
        text, df, columns = read_docx(file)
    elif file_extension == 'txt':
        text, df, columns = read_txt(file)
    elif file_extension == 'csv':
        text, df, columns = read_csv(file)
    elif file_extension in ['xlsx', 'xls']:
        text, df, columns = read_excel(file)
    else:
        return None, f"Desteklenmeyen dosya formatı: {file_extension}"

    if text is None:
        return None, f"Dosya okunamadı: {file_name}"

    # Chunking işlemi
    if df is not None and columns is not None:
        # Tablo verisi var
        chunks = create_chunks_with_columns(
            df, columns, chunk_method, chunk_size, chunk_overlap,
            separator, include_columns_in_text
        )
        # Her chunk'a dosya adı ekle
        for chunk in chunks:
            chunk['filename'] = file_name
    else:
        # Normal metin
        chunks = create_chunks(text, chunk_method, chunk_size, chunk_overlap, separator)
        # Her chunk'a dosya adı ekle
        for chunk in chunks:
            chunk['filename'] = file_name

    return chunks, None


def display_chunks(all_chunks: List[Dict]):
    """Chunk'ları görselleştir"""
    if not all_chunks:
        return

    st.markdown("### 📊 Chunk İstatistikleri")

    # Dosya bazlı istatistikler
    file_stats = {}
    for chunk in all_chunks:
        filename = chunk.get('filename', 'Bilinmeyen')
        if filename not in file_stats:
            file_stats[filename] = {'count': 0, 'total_length': 0}
        file_stats[filename]['count'] += 1
        file_stats[filename]['total_length'] += len(chunk.get('text', ''))

    # Genel istatistikler
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Toplam Chunk", len(all_chunks))

    with col2:
        st.metric("İşlenen Dosya", len(file_stats))

    with col3:
        avg_length = sum(len(chunk.get('text', '')) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        st.metric("Ort. Uzunluk", f"{avg_length:.0f} karakter")

    with col4:
        total_chars = sum(len(chunk.get('text', '')) for chunk in all_chunks)
        st.metric("Toplam Karakter", f"{total_chars:,}")

    # Dosya bazlı detaylar
    st.markdown("#### 📁 Dosya Detayları")
    for filename, stats in file_stats.items():
        with st.expander(f"📄 {filename}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunk Sayısı", stats['count'])
            with col2:
                st.metric("Toplam Karakter", f"{stats['total_length']:,}")
            with col3:
                avg = stats['total_length'] / stats['count'] if stats['count'] > 0 else 0
                st.metric("Ortalama", f"{avg:.0f} kar.")

    st.markdown("---")

    # Chunk'ları göster
    st.markdown("### 📝 Oluşturulan Chunk'lar")

    # Filtreleme seçenekleri
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Chunk'larda ara:", "")
    with col2:
        selected_file = st.selectbox(
            "📁 Dosya filtresi:",
            ["Tümü"] + list(file_stats.keys())
        )

    # Görüntüleme seçenekleri
    col1, col2, col3 = st.columns(3)
    with col1:
        show_metadata = st.checkbox("Metadata göster", value=True)
    with col2:
        show_columns = st.checkbox("Sütunları göster", value=True)
    with col3:
        show_preview_only = st.checkbox("Sadece önizleme", value=False)

    # Filtrelenmiş chunk'ları göster
    filtered_chunks = []
    for i, chunk in enumerate(all_chunks, 1):
        text = chunk.get('text', '')
        filename = chunk.get('filename', 'Bilinmeyen')

        # Filtreleme
        if search_term and search_term.lower() not in text.lower():
            continue
        if selected_file != "Tümü" and filename != selected_file:
            continue

        filtered_chunks.append((i, chunk))

    if filtered_chunks:
        st.info(f"📌 {len(filtered_chunks)} chunk gösteriliyor")

        # Sayfalama
        chunks_per_page = 10
        total_pages = (len(filtered_chunks) - 1) // chunks_per_page + 1

        if total_pages > 1:
            page = st.slider("Sayfa", 1, total_pages, 1)
        else:
            page = 1

        start_idx = (page - 1) * chunks_per_page
        end_idx = min(start_idx + chunks_per_page, len(filtered_chunks))

        for idx, chunk in filtered_chunks[start_idx:end_idx]:
            with st.expander(
                    f"Chunk {idx} - {chunk.get('filename', 'Bilinmeyen')} ({len(chunk.get('text', ''))} karakter)"):

                if show_metadata:
                    st.markdown("**📋 Metadata:**")
                    metadata_cols = st.columns(4)
                    with metadata_cols[0]:
                        st.caption(f"ID: {idx}")
                    with metadata_cols[1]:
                        st.caption(f"Dosya: {chunk.get('filename', 'N/A')}")
                    with metadata_cols[2]:
                        st.caption(f"Uzunluk: {len(chunk.get('text', ''))}")
                    with metadata_cols[3]:
                        st.caption(f"Kelime: {len(chunk.get('text', '').split())}")

                if show_columns:
                    # Sütun verilerini göster (varsa)
                    column_data = {k: v for k, v in chunk.items() if
                                   k not in ['text', 'filename', 'chunk_id', 'length']}
                    if column_data:
                        st.markdown("**🏷️ Sütun Verileri:**")
                        for col_name, col_value in column_data.items():
                            st.write(f"• **{col_name}:** {col_value}")

                st.markdown("**📝 İçerik:**")
                text_to_show = chunk.get('text', '')
                if show_preview_only and len(text_to_show) > 500:
                    text_to_show = text_to_show[:500] + "..."
                st.code(text_to_show, language=None)
    else:
        st.warning("Arama kriterlerine uygun chunk bulunamadı.")


def export_chunks(chunks: List[Dict], format: str):
    """Chunk'ları dışa aktar"""
    if format == "JSON":
        # JSON formatında tüm veriyi dahil et
        export_data = []
        for i, chunk in enumerate(chunks, 1):
            chunk_export = {
                "chunk_id": i,
                "filename": chunk.get('filename', 'unknown')
            }

            # Sütun verilerini ekle (varsa)
            for key, value in chunk.items():
                if key not in ['text', 'filename']:
                    chunk_export[key] = value

            # Text ve length ekle
            chunk_export["text"] = chunk.get('text', '')
            chunk_export["length"] = len(chunk.get('text', ''))

            export_data.append(chunk_export)

        return json.dumps(export_data, ensure_ascii=False, indent=2)

    elif format == "TXT":
        output = []
        for i, chunk in enumerate(chunks, 1):
            output.append(f"=== CHUNK {i} ===")
            output.append(f"Dosya: {chunk.get('filename', 'unknown')}")

            # Sütun verilerini ekle (varsa)
            column_data = {k: v for k, v in chunk.items() if k not in ['text', 'filename']}
            if column_data:
                output.append("Sütun Verileri:")
                for col_name, col_value in column_data.items():
                    output.append(f"  {col_name}: {col_value}")

            output.append(f"Uzunluk: {len(chunk.get('text', ''))}")
            output.append("İçerik:")
            output.append(chunk.get('text', ''))
            output.append("\n" + "=" * 50 + "\n")

        return "\n".join(output)

    elif format == "CSV":
        # CSV için düzleştirilmiş veri
        rows = []
        for i, chunk in enumerate(chunks, 1):
            row = {
                'chunk_id': i,
                'filename': chunk.get('filename', 'unknown'),
                'text': chunk.get('text', ''),
                'length': len(chunk.get('text', ''))
            }

            # Sütun verilerini ekle
            for key, value in chunk.items():
                if key not in ['text', 'filename']:
                    row[f'column_{key}'] = value

            rows.append(row)

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)


# Ana uygulama
with st.sidebar:
    st.header("⚙️ Chunking Ayarları")

    # Çoklu dosya yükleme
    uploaded_files = st.file_uploader(
        "Dosya(lar) Seçin",
        type=['txt', 'pdf', 'docx', 'csv', 'xlsx'],
        accept_multiple_files=True,
        help="Birden fazla dosya seçebilirsiniz. Desteklenen: TXT, PDF, DOCX, CSV, XLSX"
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} dosya yüklendi")
        for file in uploaded_files:
            st.caption(f"• {file.name}")

    st.markdown("---")

    # Sütun ekleme seçeneği
    include_columns_in_text = st.checkbox(
        "📊 Sütun başlıklarını text alanına ekle",
        value=False,
        help="CSV/Excel dosyalarında sütun adlarını text alanına ekler. Not: Sütunlar her zaman JSON çıktısına eklenir."
    )

    # Chunking yöntemi
    chunk_method = st.selectbox(
        "Chunking Yöntemi",
        ["Sabit Boyut", "Ayırıcı Bazlı", "Cümle Bazlı", "Satır Bazlı"],
        help="Metni nasıl böleceğinizi seçin"
    )

    # Yönteme göre parametreler
    if chunk_method == "Sabit Boyut":
        chunk_size = st.slider(
            "Chunk Boyutu (karakter)",
            min_value=100,
            max_value=2000,
            value=500,
            step=50
        )

        chunk_overlap = st.slider(
            "Chunk Örtüşmesi (karakter)",
            min_value=0,
            max_value=200,
            value=50,
            step=10
        )
        separator = "\n\n"

    elif chunk_method == "Ayırıcı Bazlı":
        separator = st.text_input(
            "Ayırıcı Karakter",
            value="\n\n",
            help="Metni bölerken kullanılacak ayırıcı"
        )

        chunk_size = st.slider(
            "Maksimum Chunk Boyutu",
            min_value=100,
            max_value=2000,
            value=500,
            step=50
        )

        chunk_overlap = st.slider(
            "Chunk Örtüşmesi (karakter)",
            min_value=0,
            max_value=200,
            value=50,
            step=10
        )

    elif chunk_method == "Satır Bazlı":
        st.info("Her satır ayrı bir chunk olacak")
        chunk_size = 500
        chunk_overlap = 0
        separator = "\n"

    else:  # Cümle Bazlı
        chunk_size = st.slider(
            "Maksimum Chunk Boyutu",
            min_value=100,
            max_value=2000,
            value=500,
            step=50
        )

        chunk_overlap = st.slider(
            "Chunk Örtüşmesi (karakter)",
            min_value=0,
            max_value=200,
            value=50,
            step=10
        )

        separator = "."

    st.markdown("---")

    # İşlem butonları
    col1, col2 = st.columns(2)

    with col1:
        process_button = st.button("🚀 Chunk'lara Böl", type="primary", use_container_width=True)

    with col2:
        clear_button = st.button("🗑️ Temizle", type="secondary", use_container_width=True)

    if clear_button:
        st.session_state.all_chunks = []
        st.session_state.processed_files = []
        st.rerun()

    if process_button:
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()

            all_chunks = []
            processed_files = []
            errors = []

            for i, file in enumerate(uploaded_files):
                status_text.text(f"İşleniyor: {file.name}")
                progress_bar.progress((i + 1) / len(uploaded_files))

                try:
                    chunks, error = process_file(
                        file, chunk_method, chunk_size, chunk_overlap,
                        separator, include_columns_in_text
                    )

                    if error:
                        errors.append(error)
                    elif chunks:
                        all_chunks.extend(chunks)
                        processed_files.append(file.name)

                except Exception as e:
                    errors.append(f"{file.name}: {str(e)}")

            progress_bar.empty()
            status_text.empty()

            # Sonuçları session state'e kaydet
            st.session_state.all_chunks = all_chunks
            st.session_state.processed_files = processed_files

            # Sonuç mesajları
            if all_chunks:
                st.success(f"✅ {len(processed_files)} dosya işlendi, {len(all_chunks)} chunk oluşturuldu!")

            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
        else:
            st.warning("Lütfen en az bir dosya yükleyin!")

    # Dışa aktarma
    if st.session_state.all_chunks:
        st.markdown("---")
        st.header("💾 Dışa Aktar")

        export_format = st.selectbox(
            "Format Seçin",
            ["JSON", "TXT", "CSV"],
            help="JSON formatı tüm sütun bilgilerini içerir"
        )

        if st.button("📥 İndir", use_container_width=True):
            export_data = export_chunks(st.session_state.all_chunks, export_format)

            if export_format == "JSON":
                file_name = "chunks.json"
                mime = "application/json"
            elif export_format == "TXT":
                file_name = "chunks.txt"
                mime = "text/plain"
            else:  # CSV
                file_name = "chunks.csv"
                mime = "text/csv"

            st.download_button(
                label=f"💾 {export_format} olarak indir",
                data=export_data,
                file_name=file_name,
                mime=mime,
                use_container_width=True
            )

# Ana içerik alanı
if st.session_state.all_chunks:
    display_chunks(st.session_state.all_chunks)
else:
    # Hoş geldin mesajı
    st.markdown("""
    ### 👋 Hoş Geldiniz!

    Bu uygulama ile:
    - 📄 **Birden fazla belgeyi** aynı anda işleyebilir
    - ✂️ **Farklı yöntemlerle** chunk'lara bölebilir
    - 📊 **CSV/Excel sütunlarını** otomatik olarak JSON'a ekleyebilir
    - 💾 **Sonuçları** detaylı JSON, TXT veya CSV formatında dışa aktarabilirsiniz

    **Başlamak için:**
    1. Sol panelden bir veya birden fazla dosya yükleyin
    2. Chunking yöntemi ve parametreleri seçin
    3. "Chunk'lara Böl" butonuna tıklayın

    ---

    #### 🎯 Özellikler:
    - **Çoklu Dosya Desteği:** Aynı anda birden fazla dosya işleyin
    - **Otomatik Sütun Tespiti:** CSV/Excel dosyalarında sütunlar otomatik algılanır
    - **Akıllı JSON Çıktısı:** Tüm sütun verileri JSON'a dahil edilir
    - **Dosya İsmi Takibi:** Her chunk hangi dosyadan geldiğini bilir

    #### 📊 JSON Çıktı Formatı:
    ```json
    {
        "chunk_id": 1,
        "filename": "veri.csv",
        "Ad": "Ahmet",
        "Soyad": "Kara",
        "Yaş": "25",
        "text": "Ahmet Kara 25",
        "length": 14
    }
    ```

    **Not:** Sütunlar her zaman JSON çıktısına eklenir. Checkbox sadece text alanına 
    sütun başlıklarının eklenmesini kontrol eder.
    """)

    # Örnek veri gösterimi
    with st.expander("📋 Örnek Kullanım Senaryosu"):
        st.markdown("""
        **Senaryo:** Müşteri veritabanı (CSV) ve ürün kataloğu (Excel) dosyalarını aynı anda işleme

        1. **customers.csv** - Müşteri bilgileri
        2. **products.xlsx** - Ürün kataloğu
        3. **policies.pdf** - Şirket politikaları

        Tüm dosyalar tek seferde yüklenir ve işlenir. Her chunk:
        - Hangi dosyadan geldiğini bilir
        - CSV/Excel için sütun verilerini korur
        - Aranabilir ve filtrelenebilir
        """)

        sample_df = pd.DataFrame({
            'MüşteriID': ['C001', 'C002', 'C003'],
            'Ad': ['Ahmet', 'Ayşe', 'Mehmet'],
            'Soyad': ['Kara', 'Demir', 'Yılmaz'],
            'Şehir': ['İstanbul', 'Ankara', 'İzmir']
        })
        st.dataframe(sample_df)
        st.caption("Bu veri işlendiğinde, her satır için sütun bilgileri JSON'a otomatik eklenir.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>📄 Gelişmiş Doküman Chunking Uygulaması v3.0 | Çoklu dosya ve gelişmiş JSON desteği</small>
</div>
""", unsafe_allow_html=True)