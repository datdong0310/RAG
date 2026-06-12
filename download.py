from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    "keepitreal/vietnamese-sbert"
)
model.save("models/vietnamese-sbert")
print("done")