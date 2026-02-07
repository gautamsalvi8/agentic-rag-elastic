from pypdf import PdfReader

def read_pdf(file_path):
    reader = PdfReader(file_path)
    content = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            content.append(page_text)

    return "\n".join(content)


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
