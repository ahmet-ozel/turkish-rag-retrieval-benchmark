import pypdf
import docx
import pandas as pd
import chardet
from typing import List, Dict, Any, Optional, Tuple
import os

class ChunkingProcessor:
    """Doküman chunking işlemlerini yöneten sınıf"""

    def __init__(self):
        self.supported_formats = ['txt', 'pdf', 'docx', 'csv', 'xlsx', 'xls']

    def detect_encoding(self, file_bytes: bytes) -> str:
        """Dosya encoding'ini tespit et"""
        result = chardet.detect(file_bytes)
        return result['encoding'] if result['encoding'] else 'utf-8'

    def read_pdf(self, file_path: str) -> Tuple[str, Optional[pd.DataFrame], Optional[List[str]]]:
        """PDF dosyasını oku"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text, None, None
        except Exception as e:
            raise Exception(f"PDF okuma hatası: {str(e)}")

    def read_docx(self, file_path: str) -> Tuple[str, Optional[pd.DataFrame], Optional[List[str]]]:
        """DOCX dosyasını oku"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text, None, None
        except Exception as e:
            raise Exception(f"DOCX okuma hatası: {str(e)}")

    def read_txt(self, file_path: str) -> Tuple[str, Optional[pd.DataFrame], Optional[List[str]]]:
        """TXT dosyasını oku"""
        try:
            with open(file_path, 'rb') as file:
                file_bytes = file.read()
                encoding = self.detect_encoding(file_bytes)
                try:
                    text = file_bytes.decode(encoding)
                except:
                    text = file_bytes.decode('utf-8', errors='ignore')
                return text, None, None
        except Exception as e:
            raise Exception(f"TXT okuma hatası: {str(e)}")

    def read_csv(self, file_path: str) -> Tuple[str, Optional[pd.DataFrame], Optional[List[str]]]:
        """CSV dosyasını oku ve sütun bilgilerini sakla"""
        try:
            df = pd.read_csv(file_path)
            text = df.to_string()
            columns = df.columns.tolist()
            return text, df, columns
        except Exception as e:
            raise Exception(f"CSV okuma hatası: {str(e)}")

    def read_excel(self, file_path: str) -> Tuple[str, Optional[pd.DataFrame], Optional[List[str]]]:
        """Excel dosyasını oku ve sütun bilgilerini sakla"""
        try:
            df = pd.read_excel(file_path)
            text = df.to_string()
            columns = df.columns.tolist()
            return text, df, columns
        except Exception as e:
            raise Exception(f"Excel okuma hatası: {str(e)}")

    def create_chunks_with_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        chunk_method: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n",
        include_columns_in_text: bool = False
    ) -> List[Dict[str, Any]]:
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

    def create_chunks(
        self,
        text: str,
        method: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ) -> List[Dict[str, Any]]:
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
                            # Çok uzun parça için kırpma
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

            # Overlap uygula
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

            # Overlap uygula
            if chunk_overlap > 0 and len(chunks) > 1:
                effective_overlap = min(max(0, chunk_overlap), size - 1)
                for i in range(1, len(chunks)):
                    prev_text = chunks[i - 1]['text']
                    prefix = prev_text[-effective_overlap:] if len(prev_text) >= effective_overlap else prev_text
                    new_text = (prefix + chunks[i]['text'])
                    chunks[i]['text'] = new_text[:size]

        return chunks

    def process_file(
        self,
        file_path: str,
        file_name: str,
        chunk_method: str,
        chunk_size: int,
        chunk_overlap: int,
        separator: str,
        include_columns_in_text: bool
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Tek bir dosyayı işle"""
        try:
            file_extension = file_name.split('.')[-1].lower()

            # Dosya türüne göre okuma
            if file_extension == 'pdf':
                text, df, columns = self.read_pdf(file_path)
            elif file_extension == 'docx':
                text, df, columns = self.read_docx(file_path)
            elif file_extension == 'txt':
                text, df, columns = self.read_txt(file_path)
            elif file_extension == 'csv':
                text, df, columns = self.read_csv(file_path)
            elif file_extension in ['xlsx', 'xls']:
                text, df, columns = self.read_excel(file_path)
            else:
                return None, f"Desteklenmeyen dosya formatı: {file_extension}"

            if text is None:
                return None, f"Dosya okunamadı: {file_name}"

            # Chunking işlemi
            if df is not None and columns is not None:
                # Tablo verisi var
                chunks = self.create_chunks_with_columns(
                    df, columns, chunk_method, chunk_size, chunk_overlap,
                    separator, include_columns_in_text
                )
                # Her chunk'a dosya adı ekle
                for chunk in chunks:
                    chunk['filename'] = file_name
            else:
                # Normal metin
                chunks = self.create_chunks(text, chunk_method, chunk_size, chunk_overlap, separator)
                # Her chunk'a dosya adı ekle
                for chunk in chunks:
                    chunk['filename'] = file_name

            return chunks, None

        except Exception as e:
            return None, str(e)
