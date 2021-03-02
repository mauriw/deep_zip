import constants

def compress(txt, window_size=10):
    masked = []
    split_txt = txt.split()
    for i in range(0, len(split_txt), window_size):
        window = split_txt[i:i + window_size]
        j, _ = max(enumerate(window), key=lambda w: len(w[1]))
        window[j] = constants.MASK
        masked += window
    return ' '.join(masked)
