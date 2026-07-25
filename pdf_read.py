from pypdf import PdfReader

reader = PdfReader("rahul-sharma.pdf")

full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

print(f"Total characters extracted: {len(full_text)}")
print("\n--- First 500 characters ---\n")
print(full_text[:2000])