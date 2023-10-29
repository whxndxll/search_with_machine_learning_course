import fasttext

model_path = '/workspace/datasets/fasttext/title_model.bin'
model = fasttext.load_model(model_path)

with open("/workspace/datasets/fasttext/top_words.txt") as input_file:
    with open("/workspace/datasets/fasttext/synonyms.csv", "w") as output_file:
        for line in input_file:
            word = line.strip()
            nns = model.get_nearest_neighbors(word)
            synonyms = [synonym for (prob, synonym) in nns if prob > 0.75]

            if len(synonyms) == 0:
                continue

            output_line = f"{word},{','.join(synonyms)}\n"
            output_file.write(output_line)
