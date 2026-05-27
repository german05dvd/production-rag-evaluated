''' Phase 0 to build the RAG Model
Chunking recursive with overlap and size respect'''

# This is the same algorithm that I transform into a function for the indexer.py script
def get(text, size, overlap) -> list[str]:
    separators = ['\n\n', '\n', '.', ' ', '']

    chunks = []
    def splitter(data, contador, llamadas):
        fragments = data.split(separators[contador])
        for i in fragments:
            if len(i) < size:
                chunks.append(i)
            else:
                llamadas += 1
                llamadas = splitter(i, contador+1, llamadas)
        return llamadas

    splitter(text, 0, 0)
    chunks = [i.strip() for i in chunks if len(i.strip()) > 10]

    merged = []
    current = ""
    for i in chunks:
        if len(current) + len(i) + 1 <= size:
            current = current + " " + i if current else i
        else:
            if current:
                merged.append(current)
            current = i
    if current:
        merged.append(current)

    final = [merged[0]] if merged else []
    for i in range(1, len(merged)):
        suffix = merged[i-1][-overlap:] if len(merged[i-1]) >= overlap else merged[i-1]
        final.append(suffix + merged[i])
    return final