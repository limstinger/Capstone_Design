# save_ner_splits.py
import json
from datasets import load_dataset, ClassLabel, Sequence, Value, Features

# ① 레이블 이름 로드
label_names = [l.strip() for l in open("labels.txt", encoding="utf-8") if l.strip()]

# ② features 스키마 정의
ner_features = Features({
    "tokens": Sequence(Value("string")),
    "tags":   Sequence(ClassLabel(names=label_names))
})

# ③ tner/fin 데이터셋 로드(split=train,validation) & 캐스팅
ds_train, ds_val = load_dataset("tner/fin", split=["train","validation"], features=ner_features)

# ④ JSON 배열(.json) 으로 저장
with open("train_ner.json", "w", encoding="utf-8") as f:
    json.dump(list(ds_train), f, ensure_ascii=False)

with open("validation_ner.json", "w", encoding="utf-8") as f:
    json.dump(list(ds_val), f, ensure_ascii=False)

print("✅ train_ner.json, validation_ner.json 생성 완료")