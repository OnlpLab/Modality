import os

ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(os.getcwd())))[0]
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))


def create_sentences(fname):
    sentences = []
    sentence = ""
    for line in fname.readlines():
        if line.strip():
            token = line.split()[0]
            sentence += token + " "
        else:
            sentences.append(sentence + "\n")
            sentence = ""
    return sentences


def collect_sentences_from_modal_not_modal(dest_path, modal_not_modal_path):
    with open(os.path.join(modal_not_modal_path, f"test_modal_not_modal_space.bmes"), "r") as tag_f:
        sentences = create_sentences(tag_f)
        with open(os.path.join(dest_path, "test.txt"), "w") as sent_f:
            for sent in sentences:
                sent_f.write(sent)

    for fold in range(5):
        for dset in ["train", "dev"]:
            filepath = os.path.join(modal_not_modal_path, f"{fold}/{dset}_modal_not_modal_space.bmes")
            with open(filepath, "r") as tag_f:
                sentences = create_sentences(tag_f)
            destination_path = os.path.join(dest_path, f"{dset}_{fold}.txt")
            with open(destination_path, "w") as sent_f:
                for sent in sentences:
                    sent_f.write(sent)


if __name__ == "__main__":
    dest_path = os.path.join(ROOT_DIR, "data", "GME", "cross_validation", "sentences")
    modal_not_modal_path = os.path.join(ROOT_DIR, "data", "GME", "modal_not_modal")
    
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    collect_sentences_from_modal_not_modal(dest_path, modal_not_modal_path)
    print("DONE!")