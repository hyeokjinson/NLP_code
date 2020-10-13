with open('wiki_preprocessed.txt', 'w', encoding='utf-8') as fw:
    with open('wikiAA.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            if line.startswith('<doc') or line.startswith('</doc>') or len(line.strip()) == 0:
                continue

            fw.write(line)

print('done')