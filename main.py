import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os,sys,torch,subprocess
MODEL="./models/qwen3.5-4b"
PRIV_INPUT="./data/private_test.csv"
PUBL_INPUT="./data/public_test.csv"
OUTPUT="./output/pred.csv"

"""
    do e lười chuyển sang tiếng việt thường xuyên để viết log nên
    e viết luôn bằng tiếng anh :)) - khanhdzvcl
"""

def loadDataFrameCSV(path):
    if not os.path.exists(path):
        print(f"[error] {path} not exists")
        sys.exit(1)
    print(f"[log] start reading {path}")
    data=pd.read_csv(path)
    col_map={}
    for col in data.columns:
        col_lower = col.lower()
        if col_lower in ['id', 'q_id','question_id','questions_id']:
            col_map[col] = 'id'
        elif col_lower in ['question','questions','queries','query', 'cau_hoi']:
            col_map[col] = 'question'
        elif col_lower in ['context', 'text', 'document', 'noi_dung','data']:
            col_map[col] = 'context'
        else:col_map[col]=col_lower
    data=data.rename(columns=col_map)
    if 'id' not in data.columns:
        data['id']=data.index
    if 'question' not in data.columns:
        print("[error] could not find question")
        print(f" available data frame {list(data.columns)}")
        sys.exit(1)
    print(f"[log] completed reading {path},with {len(data)} rows")
    return data
def processAnswer(path):
    if os.path.exists(path):
        print(f"[log] loading data from {path}")
        data_frame=loadDataFrameCSV(path)
        print(f"[log] loading tokenizer and model")
        tokenizer=AutoTokenizer.from_pretrained(MODEL)
        model=AutoModelForCausalLM.from_pretrained(
            MODEL,
             torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        choice_tokens=["A","B","C","D"]
        choice_ids=[tokenizer.encode(choice)[-1] for choice in choice_tokens]
        results=[]
        print("[log] starting prompting qwen...")
        for _, row in tqdm(data_frame.iterrows(), total=len(data_frame)):
            print(f"[log] prompting question {row['id']}")
            prompt_text=(
                    f"Câu hỏi: {row['question']}\n"
                    f"Ngữ cảnh: {row['context']}\n"
                    "Hãy chọn đáp án phù hợp nhất với câu hỏi:\n"
                    f"A:{row['a']}\n"
                    f"B:{row['b']}\n"
                    f"C:{row['c']}\n"
                    f"D:{row['d']}\n"
                    "Chỉ ghi một chữ cái(ví dụ: A)"
                    )
            messages = [
                {"role": "system", "content": "You are a precise multiple-choice testing assistant. You only answer with a single letter."},
                {"role": "user", "content": prompt_text}
            ]
            formatted_prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs=model(**inputs)
                next_token_logits=outputs.logits[0,-1,:]
                target_logits=next_token_logits[choice_ids]
                best_choice_idx=torch.argmax(target_logits).item()
                predicted_letter=choice_tokens[best_choice_idx]
            results.append({
                "qid":row['id'],
                "answer":predicted_letter
            })
        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
        prediction_df=pd.DataFrame(results);
        prediction_df.to_csv(OUTPUT,index=False)
        print(f"[log] sent result to {OUTPUT}")
    else : 
        print(f"[error] test file ({path}) not found")
        sys.exit(1)
if __name__=="__main__":
    if not os.path.exists(MODEL):
        print("[log] intializing model folder")
        print(f"[log] downloading model in {MODEL}")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        subprocess.run(f"git lfs install && git clone https://huggingface.co/Qwen/Qwen3.5-4B {MODEL}",shell=True,check=True) 
        print("[log] completed downloading model")
    if not inp or inp=='y':
        processAnswer(PRIV_INPUT)
        processAnswer(PUBL_INPUT)
"""
    lm sao ngta viết comment thường xuyên đc vậy?
    với lại e thấy comment trong python trông rất xấu
"""
