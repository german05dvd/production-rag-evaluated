''' Phase 5 to build the RAG Model
Evaluate RAG: Retrieval + Generation (with diagnostics) '''

from indexer import load_model
from retriever import extract, retrieve
from generator import generate, build_prompt

model = load_model()
collection = extract()

questions = [
    # 1. Trivia
    {'question': 'According to the First Citizen, who is the "chief enemy to the people"?',
     'keywords': ['Caius', 'Marcius', 'chief', 'enemy'],
     'answer': 'Caius Marcius'},
    # 2. Abstraction
    {'question': "Explain the application of Menenius's parable about the belly.",
     'keywords': ['senators', 'Rome', 'mutinous', 'members'],
     'answer': 'The senators of Rome are this good belly'},
    # 3. Boolean
    {'question': 'Does Volumnia prefer her son to die nobly for his country rather than live without action?',
     'keywords': ['die', 'nobly', 'action', 'rather'],
     'answer': 'Yes'},
    # 4. Trivia 
    {'question': 'What new name is granted to Caius Marcius after his victory at Corioli?',
     'keywords': ['Coriolanus', 'Caius', 'Marcius'],
     'answer': 'CORIOLANUS!'},
    # 5. Trivia
    {'question': 'Who does King Richard III suborn to kill the young princes in the Tower?',
     'keywords': ['Tyrrel', 'Dighton', 'Forrest'],
     'answer': 'Tyrrel'},
    # 6. Specific
    {'question': 'What is the initial punishment King Richard II gives to Henry Bolingbroke?',
     'keywords': ['banish', 'banishment', 'summers'],
     'answer': "Banish"},
    # 7. Condition
    {'question': 'What condition does Baptista Minola set before his younger daughter Bianca can marry?',
     'keywords': ['Katharina', 'elder', 'wed', 'first'],
     'answer': 'Husband for the elder'},
    # 8. Contextual
    {'question': "According to the character 'Time', how many years pass during the gap in The Winter's Tale?",
     'keywords': ['sixteen', 'years', 'slide'],
     'answer': "Sixteen"},
    # 9. Trivia
    {'question': "What was Prospero's title before he was heaved from Milan?",
     'keywords': ['Duke', 'Milan', 'prince', 'power'],
     'answer': 'Duke of Milan'},
    # 10. Boolean 
    {'question': 'Does King Henry VI believe his title to the crown is better than that of the house of York?',
     'keywords': ['title', 'good', 'better'],
     'answer': "Yes"}
]

k = 10
keyword_scores = 0
answer_scores = 0
substring_hits = 0
generation_hits = 0

print(f"{'Question':<40} {'Keyw':<5} {'AnsRet':<8} {'GenStr':<7} {'GenKey':<6}")

for item in questions:
    retrieved = retrieve(item["question"], model, collection, k)
    documents = retrieved["documents"][0]

    kw_hit = any(any(key.lower() in doc.lower() for key in item["keywords"]) for doc in documents)
    answer = item["answer"].lower()
    ans_hit = any(answer in doc.lower() for doc in documents)

    keyword_scores += kw_hit
    answer_scores += ans_hit

    prompt = build_prompt(item["question"], documents)
    generated = generate(prompt)
    lower = generated.lower()

    hit = answer in lower
    substring_hits += hit

    keywords = all(key.lower() in lower for key in item["keywords"])
    generation_hits += keywords
    print(f"{item['question'][:30]:<40} {'ok' if kw_hit else 'bad':<5} {'ok' if ans_hit else 'bad':<8} {'ok' if hit else 'bad':<7} {'ok' if keywords else 'bad':<6}")

    print(f"\n Question: {item['question']}")
    print(f" Expected: {item['answer']}")
    print(f" Generated: {generated}\n")

total = len(questions)
print(f"\n\nRecall@{k} (keywords): {keyword_scores}/{total} = {keyword_scores/total:.2f}")
print(f"Recall@{k} (respuesta exacta): {answer_scores}/{total} = {answer_scores/total:.2f}")
print(f"Exact generation (substring): {substring_hits}/{total} = {substring_hits/total:.2f}")
print(f"Flexible generation (keywords): {generation_hits}/{total} = {generation_hits/total:.2f}")