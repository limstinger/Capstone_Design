import os
import numpy as np
import re, pandas as pd, torch
from gensim.models import FastText

if not hasattr(np, "dtypes"):
    np.dtypes = np.dtype

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, TrainingArguments, Trainer
)
from datasets import Dataset
from evaluate import load  # 변경된 함수명
metric = load("accuracy")

from transformers import EarlyStoppingCallback
from transformers import __version__ as tv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 데이터 파일 경로
CSV_PATH = os.path.join(BASE_DIR, "models", "data", "articles_export.csv")
XSLX_PATH = os.path.join(BASE_DIR, "models", "data", "KOSELF(2.0).xlsx")

# output_dir
OUTPUT_DIR = os.path.join(BASE_DIR, "finance_project", "finetuned")



# 1) 데이터 로드
df = pd.read_csv(CSV_PATH)
headlines = df['headline'].dropna().tolist()


# 2) KOSELF 사전 로드
xls     = pd.ExcelFile(XSLX_PATH)
pos_lex = set(pd.read_excel(xls, 'positive').iloc[:,0].dropna().str.strip())
neg_lex = set(pd.read_excel(xls, 'negative').iloc[:,0].dropna().str.strip())

# 3) 토큰화 및 FastText 학습(사전 확장)
def tokenize(text):
    tokens = re.findall(r'[가-힣]+', text)
    return [t for t in tokens if len(t)>1]
corpus = [tokenize(h) for h in headlines]
ft = FastText(sentences=corpus, vector_size=100, window=5, min_count=5)
for w in list(pos_lex):
    pos_lex |= {sim for sim,_ in ft.wv.most_similar(w, topn=3)}
for w in list(neg_lex):
    neg_lex |= {sim for sim,_ in ft.wv.most_similar(w, topn=3)}

# 4) 자동 라벨링
def auto_label(headline):
    for w in pos_lex:
        if w in headline:
            return 2   # 긍정
    for w in neg_lex:
        if w in headline:
            return 0   # 부정
    return 1          # 중립

labels = [auto_label(h) for h in headlines]

# 5) HF Dataset 생성 및 토크나이징
ds = Dataset.from_dict({'text': headlines, 'label': labels})
tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-FinBert-SC')
def preprocess(ex):
    return tokenizer(ex['text'], truncation=True, padding='max_length', max_length=128)
ds = ds.map(preprocess, batched=True)
ds = ds.train_test_split(test_size=0.2)

# 6) 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    'snunlp/KR-FinBert-SC', num_labels=3
)

# 7) Trainer 설정
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    # 로깅
    logging_steps=50,
    logging_strategy='steps',

    eval_strategy='steps',
    eval_steps=100,

    save_strategy='steps',
    save_steps=100,

    # 학습률
    learning_rate=2e-5,

    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True
)

from torch.nn import CrossEntropyLoss

# ① 클래스별 가중치 계산 (labels 리스트는 이미 생성된 상태여야 함)
counts = np.bincount(labels)
class_weights = torch.tensor(1.0/counts, dtype=torch.float).to(model.device)

# ② compute_loss 함수 정의
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    loss_fct = CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(outputs.logits, labels)
    return (loss, outputs) if return_outputs else loss


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# 8) 파인튜닝 실행
trainer.train()

# 9) 최종 모델 저장
trainer.save_model(OUTPUT_DIR)
