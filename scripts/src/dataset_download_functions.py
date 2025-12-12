def load_classes(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]    
