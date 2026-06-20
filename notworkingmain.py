
"""
    do e lười chuyển sang tiếng việt thường xuyên để viết log nên
    e viết luôn bằng tiếng anh :)) - khanhnotbruh
"""
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
import os,sys,torch,subprocess
#MODEL = "Qwen/Qwen3.5-4B"
MODEL="./models/qwen3.5-4b"
PRIV_INPUT="./data/private-test.json"
PUBL_INPUT="./data/public-test.json"
OUTPUT="./output/pred.csv"

def loadDataJson(path):
    if not os.path.exists(path):
        print(f"[error] {path} not exists")
        sys.exit(1)
    print(f"[log] start reading {path}")
    data=pd.read_json(path)
    col_map={}
    for col in data.columns:
        col_lower = col.lower()
        if col_lower in ['id', 'qid','questions_id']:
            col_map[col] = 'id'
        elif col_lower in ['question','questions','queries','query', 'cau_hoi']:
            col_map[col] = 'question'
        else:col_map[col]=col_lower
    data=data.rename(columns=col_map)
    if 'choices' in data.columns:
        print("[log] parsing answer")
        choices=pd.DataFrame(data['choices'].tolist(), index=data.index);
        choices=choices.rename(columns={0: 'a', 1: 'b', 2: 'c', 3: 'd'})
        data = pd.concat([data, choices], axis=1)
    else:
        print("[error] choices not found")
        sys.exit(1)
    if 'id' not in data.columns:
        data['id']=data.index
    if 'question' not in data.columns:
        print("[error] could not find question")
        print(f" available data: {list(data.columns)}")
        sys.exit(1)
    for choice in ['a', 'b', 'c', 'd']:
        if choice not in data.columns:
            print(f"[error] missing choices column: '{choice}'")
            sys.exit(1)
    print(f"[log] completed reading {path},with {len(data)} rows")
    return data
def initModel(model_name):
    print(f"[log] loading tokenizer and model")
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"

    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return tokenizer,model
def processAnswer(data,tokenizer,model,batch_size=4):
    if not data.empty:
        choice_tokens=["A","B","C","D"]
        choice_ids=[tokenizer.encode(choice)[-1] for choice in choice_tokens]
        results=[]

        prompts=[]
        qids=[]
        print("[log] intializing batches")
        for _,row in data.iterrows():
            prompts.append(tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a precise multiple-choice testing assistant. You only answer with a single letter."},
                    {"role": "user", "content": (
                            "Bối cảnh văn bản:\n"
                            f"{row['question']}\n"
                            "Hãy chọn đáp án phù hợp nhất với câu hỏi:\n"
                            f"A:{row['a']}\n"
                            f"B:{row['b']}\n"
                            f"C:{row['c']}\n"
                            f"D:{row['d']}\n"
                            "Yêu cầu phản hồi: Bạn chỉ được ghi duy nhất một chữ cái đại diện cho đáp án đúng (ví dụ: A hoặc B hoặc C hoặc D). Tuyệt đối không giải thích gì thêm."
                        )
                    }
                ],
                tokenize=False,add_generation_prompt=True
                )
            )
            qids.append(row['id'])
        print("[log] starting to prompts qwen")
        for i in tqdm(range(0, len(prompts), batch_size)):
            cprompts=prompts[i:i+batch_size]
            cqid=qids[i:i+batch_size]
            inputs=tokenizer(cprompts,return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad():
                outputs=model(**inputs)
                best=torch.argmax(outputs.logits[:,-1,:][:,choice_ids], dim=-1).cpu().tolist()
            for qid,idx in zip(cqid,best):
                results.append({
                    "qid": qid,
                    "answer": choice_tokens[idx]
                })
        print(f"[log] completed processing result")
        return pd.DataFrame(results);
    else : 
        print(f"[error] data not found")
        sys.exit(1)
if __name__=="__main__":
    if not os.path.exists(MODEL):
        print("[log] intializing model folder")
        print(f"[log] downloading model in {MODEL}")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        subprocess.run(f"git lfs install && git clone https://huggingface.co/Qwen/Qwen3.5-4B {MODEL}",shell=True,check=True) 
        print("[log] completed downloading model")
    tokenizer,model=initModel(MODEL)
    final_answer=[]
    if os.path.exists(PUBL_INPUT):
        final_answer.append(processAnswer(loadDataJson(PUBL_INPUT),tokenizer,model))
    if os.path.exists(PRIV_INPUT):
        final_answer.append(processAnswer(loadDataJson(PRIV_INPUT),tokenizer,model))
    if final_answer:
        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
        print(f"[log] printing answer to {OUTPUT}")
        pd.concat(final_answer,ignore_index=True).to_csv(OUTPUT,index=False)
        print(f"[log] completed printing answer")
    else:
        print(f"[error] found no files in {PRIV_INPUT} and {PUBL_INPUT}")
"""
    lm sao ngta viết comment thường xuyên đc vậy?
    với lại e thấy comment trong python trông rất xấu
"""
