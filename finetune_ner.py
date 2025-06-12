import os
import torch
from datasets import load_dataset, ClassLabel, Sequence, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# 1) 설정
MODEL    = "kakaobank/kf-deberta-base"
OUT_DIR  = "outputs/kf-deberta-fin-ner"
device   = "cuda" if torch.cuda.is_available() else "cpu"

# 2) 원본 tner/fin 데이터셋 불러오기
raw_ds = load_dataset("tner/fin")

# 3) 레이블 이름 직접 정의 (tner/fin tags 0~7 순서에 맞춤)
label_names = ["O","B-PER","B-LOC","B-ORG","B-MISC","I-PER","I-LOC","I-ORG"]

# 4) Features 재정의: tokens, tags 컬럼 타입 지정
ner_features = Features({
    "tokens": Sequence(Value("string")),
    "tags":   Sequence(ClassLabel(names=label_names))
})
ds = raw_ds.map(
    lambda batch: {"tokens": batch["tokens"], "tags": batch["tags"]},
    batched=True,
    features=ner_features,
)

# 5) 토크나이저 & 모델 로드 (토크나이저 신뢰, 토치 디바이스 이동)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model     = AutoModelForTokenClassification.from_pretrained(
    MODEL,
    num_labels=len(label_names),
    id2label={i: n for i,n in enumerate(label_names)},
    label2id={n: i for i,n in enumerate(label_names)},
    trust_remote_code=True
).to(device)

# 6) 토큰화 및 라벨 얼라인 함수
def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=128
    )
    labels = []
    for i, label_seq in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned  = []
        previous = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)  # special token
            elif wid != previous:
                aligned.append(label_seq[wid])
            else:
                # subword일 땐 B- → I- 처리
                orig = label_names[label_seq[wid]]
                if orig.startswith("B-"):
                    aligned.append(label_names.index(orig.replace("B-", "I-")))
                else:
                    aligned.append(label_seq[wid])
            previous = wid
        labels.append(aligned)
    tokenized["labels"] = labels
    return tokenized

# 7) 데이터셋 토크나이징
tokenized_ds = ds.map(
    tokenize_and_align,
    batched=True,
    remove_columns=["tokens", "tags"]
)

# 8) Trainer 준비
data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=16,
    learning_rate=3e-5,
    num_train_epochs=5,      # 에폭을 늘립니다
    eval_strategy="steps",    # <-- evaluation_strategy → eval_strategy
    eval_steps=500,           # (이름은 그대로)
    save_strategy="steps",    # 최신 이름이지만, 구버전에서도 동작할 수 있으니
    save_steps=500,           # 문제가 계속되면 이 줄만 살리고 위 두 줄 삭제
    save_total_limit=2,
    do_train=True,
    do_eval=True,
    overwrite_output_dir=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 9) 학습 및 모델 저장
trainer.train()
trainer.save_model(OUT_DIR)