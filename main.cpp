/*
 * -- random comment --
 * "im going to be autistic" -17/6/2026
 * "lý do tại s e ko có log cho parsing hay là read file
 * vì nó chạy quá nhanh :))"-18/6/2026-
 * "pls check line 210"-19/6/2026-
 * "dcm s t ko đặt tên cho cái dictionary là pos ;-;" -20/6/2026-
 * "i hate goto"-21/6/2026-
 * "h lại phải chuyển từ json.h sang simdjson ;-;"-21/6/2026-
 * "dcm sao lại có tận 10 lựa chọn?????"-22/6/2026-
 * -- road map --
 *  parses->tokenize->batch->decode->model (super simple)
 *  mất 3 ngày lang thang ở python và docker
 *  1d dành cho parse: debug 3d->4 ngày
 *  3d dành cho model: debug 2d->5 ngày
 *  0.5d viết lại parser:debug 0.1d->1 ngày
 *  1d viết lại hết decoder:debug 1d->2 ngày
 * -- note--
 *  code này được viết 90% bởi e (có mỗi cái bảng ở dòng ~251)
 *  có sự tham khảo của gemini và clade
 *  pls be impressed ;-;
*
 *  dự án bắt đầu từ ngày 10/6/2026
 *  ban đầu đc viết bằng python
 *  nhưng mà nó chạy chậm vl
 *
 *   stat hackathon|grep Birth 
 *  -> Birth: 2026-06-10 20:37:01.986511491 +0700
 *
 *  lúc đếy e còn chx bt j về llama.cpp
 *  cảm ơn chị google đã tạo ra google collab và google gemini
 *  --top tier ascii art --
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⡀ ♥VN⠀⠀⢀⣤⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡖⠀⠀⠀⠀⢲⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣀⣴⣦⣀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 * ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
 */
#include "simdjson.h"
#include "llama.h"
#include <alloca.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
using namespace std;
#define MODEL "./models/Qwen3.5-9B-Q8_0.gguf"// dangerous 10gb file
#define TEST "./models/Qwen3.5-0.8B-Q8_0.gguf" 
#define PUBLIC "./data/public-test.json"
#define PRIVATE "./data/private-test.json"
#define OUTPUT "./output/pred.csv"

struct data_s{
  char*data;
  char***dictionary;
  int total_question,total_option,total_char;
};
struct token_s{
  llama_token*data;
  llama_token**dictionary;
  int total_token,total_number;
};
struct model_s{
  struct llama_model*model;
  struct llama_context*ctx;
  struct llama_sampler*smpl;
};
void cleanUpData(char**data,char****dictionary,int*total_question,int*total_char){
  if(data&&*data){free(*data);*data=0;}
  if(dictionary&&*dictionary){
    for(int a=0;a<(*total_question);a++)free((*dictionary)[a]);
    free(*dictionary); *dictionary=0;
  }
  if(total_question)*total_question=0;
  if(total_char)*total_char=0;
}
void cleanUpToken(llama_token**data,llama_token***dictionary,int*size_token,int*size_number){
  if(data&&*data){free(*data);*data=0;}
  if(dictionary&&*dictionary){free(*dictionary);*dictionary=0;}
  if(size_token)*size_token=0;
  if(size_number)*size_number=0;
}
void cleanUpModel(struct llama_model**model,struct llama_context**ctx,struct llama_sampler**smpl){
  if(model&&*model){llama_model_free(*model);*model=0;}
  if(ctx&&*ctx){llama_free(*ctx);*ctx=0;}
  if(smpl&&*smpl){llama_sampler_free(*smpl);*smpl=0;}
}

char*readFile(const char*path,size_t*size_out){
  FILE*input=fopen(path,"rb");
  if(!input)return 0;
  fseek(input,0,SEEK_END);
  size_t i_size=ftell(input);
  rewind(input);
  char*buffer=(char*)malloc(i_size+1);
  if(!buffer){
    printf("[error] failed allocating buffer for reading file");
    fclose(input);
    return buffer;
  }
  size_t bsize=fread(buffer,1,i_size,input);
  if(bsize!=i_size){
    free(buffer);
    buffer=0;
    printf("[error] unexpected value difference between bsize (%d) and i_size (%d)",bsize,i_size);
    fclose(input);
    return buffer;
  }
  buffer[bsize]='\0';
  fclose(input);
  *size_out=i_size;
  return buffer;
}
#include <string>//;-; cant avoid using this
struct data_s parsingDataJson(char*raw[],int raw_size[],int size){
  if(!raw){
    printf("[error] pass null pointer to parse??");
    return{};
  }
  int total_size=0;
  for(int i=0;i<size;i++){
    total_size+=raw_size[i];
    if(!raw[i]){
      printf("[error] passing null pointer as content to parse?");
      return {};
    }
  }
  char*data=(char*)malloc(total_size+1);
  if(!data){
    printf("[log] failed allocating space while parsing");
    cleanUpData(&data,0,0,0);
    return {};
  }
  simdjson::ondemand::parser parser;
  int total_option=0;
  int total_question=0;
  for (int i=0;i<size;i++){
    simdjson::padded_string_view padded_view(raw[i],raw_size[i],raw_size[i]+simdjson::SIMDJSON_PADDING);
    simdjson::ondemand::document doc=parser.iterate(padded_view);
    for(simdjson::ondemand::object obj:doc.get_array()){
      simdjson::ondemand::array choices=obj["choices"].get_array();
      int c=choices.count_elements();
      if(c>total_option)total_option=c;
      total_question++;
    }
  }
  total_option+=2;
  char***dictionary=(char***)malloc(total_question*sizeof(char**));
  if(!dictionary){
    printf("[log] failed allocating space while parsing");
    cleanUpData(&data,&dictionary,0,0);
    return {};
  }
  for(int i=0;i<total_question;i++){
    dictionary[i]=(char**)malloc(total_option*sizeof(char*));
    if(!dictionary[i]){
      printf("[log] failed allocating space while parsing");
      cleanUpData(&data,&dictionary,0,0);
      return {};
    }
    for(int j=0;j<total_option;j++)dictionary[i][j]=0;
  }
  char*write_ptr=data;
  int qidx=0;
  for(int i=0;i<size;i++){
    simdjson::padded_string_view padded_view(raw[i],raw_size[i],raw_size[i]+simdjson::SIMDJSON_PADDING);
    simdjson::ondemand::document doc=parser.iterate(padded_view);
    simdjson::ondemand::array root_array=doc.get_array();
    for(simdjson::ondemand::object obj:root_array){
      int oidx=0;
      auto write=[&](auto&&data) {
        dictionary[qidx][oidx]=write_ptr;
        string_view view=data.get_string();
        memcpy(write_ptr,view.data(),view.size());
        write_ptr+=view.size();
        oidx++;
      };
      write(obj["qid"]);
      write(obj["question"]);
      simdjson::ondemand::array choices=obj["choices"].get_array();
      for(auto choice:choices)write(choice);
      qidx++;
    }
  }
  *write_ptr='\0';
  return {data,dictionary,total_question,total_option,(int)(write_ptr-data)};
}

// i am bored with using goto
// pls input a valid one : [l,r]<n;
struct token_s tokenizeRange(llama_model*model,const char*prompt_pos[],int len[],int size,int l,int r){
  if(l>r||r>=size){
    printf("[log] invalid input for tokenizeRange function");
    return {};
  }
  int total_size=0;
  for(int i=l;i<=r;i++)total_size+=len[i];
  int max_token=total_size+(16*(r-l+1));
  llama_token*token=(llama_token*)malloc(max_token*sizeof(llama_token));
  llama_token**dictionary=(llama_token**)malloc((r-l+1)*sizeof(llama_token*));
  if(!token||!dictionary){
    printf("[error] allocation failed while initializing space for tokenizing");
    cleanUpToken(&token,&dictionary,0,0);
    return {};
  }
  llama_token*write_ptr=token;
  for(int i=0;i<(r-l+1);i++)dictionary[i]=0;
  int used_size=0;
  for(int i=l;i<=r;i++){
    const struct llama_vocab* vocab = llama_model_get_vocab(model);
    int cur_size=llama_tokenize(vocab,prompt_pos[i],len[i],write_ptr,max_token-used_size,false,true);
    if(cur_size<0){
      printf("[error] failed to tokenize");
      cleanUpToken(&token,&dictionary,0,0);
      return{};
    }
    dictionary[i-l]=write_ptr;
    write_ptr+=cur_size;
    used_size+=cur_size;
  }
  total_size=write_ptr-token;
  return {token,dictionary,r-l+1,total_size};
}
void appendToBatch(struct llama_batch*batch,llama_token*t,int*pos,int size,int sseq_id,int eseq_id){
  int off=batch->n_tokens;
  int bsize=eseq_id-sseq_id+1;
  memcpy(batch->token+off,t,size*sizeof(*t));
  for(int i=0;i<size;i++){
    int c=off+i;
    batch->pos[c]=(*pos)++;
    batch->n_seq_id[c]=bsize;
    for(int j=0;j<bsize;j++)batch->seq_id[c][j]=sseq_id+j;
    batch->logits[c]=false;
  }
  if(size>0)batch->logits[off+size-1]=true;
  batch->n_tokens+=size;
}
token_s initSharedPromptToken(struct model_s*model,int total_option){
  int size=total_option+2;
  const char**prompt=(const char**)alloca(size*sizeof(char*));
  prompt[0]="<|im_start|>system\nBạn là một trợ lý ảo vô cùng chính xác, chuyên xử lý bài thi trắc nghiệm. Nhiệm vụ của bạn là đọc câu hỏi và các lựa chọn được cung cấp, sau đó phân tích để tìm ra đáp án đúng nhất.\nYÊU CẦU BẮT BUỘC: Chỉ được phép trả về duy nhất một chữ cái đại diện cho đáp án đúng (A đến Z). Không giải thích, không thêm khoảng trắng, không lặp lại câu hỏi.<|im_end|>\n<|im_start|>user\n\nCác lựa chọn:\n";
  prompt[size-1]="\nĐáp án đúng là:<|im_end|>\n<|im_start|>assistant\n";
  int *strsize=(int*)alloca(size*sizeof(int));
  strsize[0]=strlen(prompt[0]);
  strsize[size-1]=strlen(prompt[size-1]);
  // ai generated table ;-;
  const char* options[26] = {
    "\nA: ", "\nB: ", "\nC: ", "\nD: ", "\nE: ", "\nF: ", "\nG: ", "\nH: ", "\nI: ", "\nJ: ", "\nK: ", "\nL: ", "\nM: ",
    "\nN: ", "\nO: ", "\nP: ", "\nQ: ", "\nR: ", "\nS: ", "\nT: ", "\nU: ", "\nV: ", "\nW: ", "\nX: ", "\nY: ", "\nZ: "
  };
  for(int i=1;i+1<size;i++){
    prompt[i]=options[i-1];
    strsize[i]=4;
  }
  return tokenizeRange(model->model,prompt,strsize,size,0,size-1);
};
void processRangeQuestion(struct data_s*data,struct model_s*model,struct token_s*shared_token,int sqidx,int eqidx,int*seq_pos,FILE*out){
  if(sqidx>eqidx)return;
  int max_option=data->total_option-2;
  llama_token**shared_pos=(llama_token**)alloca((max_option+3)*sizeof(int*));
  for(int i=0;i<max_option+2;i++)shared_pos[i]=shared_token->dictionary[i];
  shared_pos[max_option+2]=shared_token->data+shared_token->total_number;
  int total_question=eqidx-sqidx+1;
  int shared_size=shared_pos[1]-shared_pos[0];
  int segment_size=((eqidx+1==data->total_question)?data->data+data->total_char:data->dictionary[eqidx+1][0])-data->dictionary[sqidx][0];//??
  int max_token=segment_size+total_question*(shared_pos[max_option+2]-shared_pos[max_option+1])+shared_size;
  int max_batch=llama_n_batch(model->ctx);
  if(sqidx==eqidx&&max_token>max_batch){
    int qid_size = data->dictionary[sqidx][1]-data->dictionary[sqidx][0];
    printf("[warning] encountered a very big question (%.*s:%d tokens) relative to your small ass vram\n",qid_size,data->dictionary[sqidx][0],max_token);
    fprintf(out, "%.*s,C\n",qid_size,data->dictionary[sqidx][0]);
    fflush(out);
    return;
  }
  //recursion stuff
  if(max_batch>=max_token){
    printf("[log] processing %d question\n",total_question);
    struct llama_batch batch=llama_batch_init(max_token,0,total_question);
    batch.n_tokens=0;
    int *logit_idx=(int*)alloca(total_question*sizeof(int));
    int pos=0;
    appendToBatch(&batch,shared_pos[0],&pos,shared_size,*seq_pos,*seq_pos+total_question-1);
    for(int bidx=0;bidx<total_question;bidx++){
      pos=shared_size;
      int qidx=bidx+sqidx,ci;
      const char**prompt_pos=(const char**)alloca((max_option+2)*sizeof(int*));
      for(ci=0;ci<=max_option;ci++){
        const char*c=data->dictionary[qidx][ci+1];
        if(!c)break;
        prompt_pos[ci]=c;
      }
      prompt_pos[ci]=(qidx+1==data->total_question)?(data->data+data->total_char):data->dictionary[qidx+1][0];
      int*strsize=(int*)alloca(ci*sizeof(int));
      for(int i=0;i<ci;i++){
        strsize[i]=prompt_pos[i+1]-prompt_pos[i];
      }
      struct token_s tk=tokenizeRange(model->model,prompt_pos,strsize,ci,0,ci-1);
      llama_token**tk_pos=(llama_token**)alloca((ci+1)*sizeof(llama_token*));
      for(int i=0;i<ci;i++)tk_pos[i]=tk.dictionary[i];
      tk_pos[ci]=tk.data+tk.total_number;
      int seq_id=*seq_pos+bidx;
      appendToBatch(&batch,tk_pos[0],&pos,tk_pos[1]-tk_pos[0],seq_id,seq_id);
      for(int i=1;i<ci;i++){
        int j=i+1;
        appendToBatch(&batch,shared_pos[i],&pos,shared_pos[j]-shared_pos[i],seq_id,seq_id);
        appendToBatch(&batch,tk_pos[i],&pos,tk_pos[j]-tk_pos[i],seq_id,seq_id);
      }
      appendToBatch(&batch,shared_pos[max_option+1],&pos,shared_pos[max_option+2]-shared_pos[max_option+1],seq_id,seq_id);
      cleanUpToken(&tk.data,&tk.dictionary,&tk.total_token,&tk.total_number);
      logit_idx[bidx]=batch.n_tokens-1;
    }
    *seq_pos+=total_question;
    // decode
    int res=llama_decode(model->ctx,batch);
    if(!res){
      for(int bidx=0;bidx<total_question;bidx++){
        int qidx=sqidx+bidx;
        llama_token prediction=llama_sampler_sample(model->smpl,model->ctx,logit_idx[bidx]);
        char out_buf[64];
        const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
        int res_len=llama_token_to_piece(vocab,prediction,out_buf,sizeof(out_buf),0,false);
        char ans='C';
        for (int i = 0; i < res_len; i++) {
          if (out_buf[i]>='A'&&out_buf[i]<='Z'){ans=out_buf[i];break;}
          if (out_buf[i]>='a'&&out_buf[i]<='z'){ans=out_buf[i]&~32;break;}
        }
        int qid_size=data->dictionary[qidx][1]-data->dictionary[qidx][0];
        fprintf(out,"%.*s,%c\n",qid_size,data->dictionary[qidx][0],ans);
      }
      fflush(out);
    }else{
      printf("[error] decode execution failed for range %d to %d. defaulting to 'C'\n", sqidx, eqidx);
      for(int i=sqidx;i<=eqidx;i++) {
        int qid_size=data->dictionary[i][1]-data->dictionary[i][0];
        fprintf(out,"%.*s,C\n",qid_size,data->dictionary[i][0]);
      }
      fflush(out);
    }
    llama_batch_free(batch);
  }else{
    int mid=sqidx+(total_question-1)/2;
    printf("[log] divided into two branch: from %d to %d and %d to %d\n",sqidx,mid,mid+1,eqidx);
    processRangeQuestion(data,model,shared_token,sqidx,mid,seq_pos,out);
    processRangeQuestion(data,model,shared_token,mid+1,eqidx,seq_pos,out);
  }
}

struct model_s initializeModel(const char*model_path,bool*init){
  struct llama_context*ctx=0;
  struct llama_model*model=0;
  struct llama_sampler*smpl=0;
  printf("[log] initializing llama backend\n");
  if(!(*init)){
    llama_backend_init();
    *init=1;
  }
  printf("[log] completed initializing llama backend\n");
  printf("[log] loading parameters\n");
  llama_model_params model_params=llama_model_default_params();
  model_params.n_gpu_layers = 99; //<- use this if you have strong af gpu
  struct llama_context_params context_params=llama_context_default_params();
  context_params.n_batch=32768;
  context_params.n_ctx=32768;
  context_params.n_seq_max=256;
  context_params.kv_unified=true;
  printf("[log] completed loading parameters\n");
  printf("[log] loading model: %s\n",model_path);
  model=llama_model_load_from_file(model_path,model_params);
  if(!model){
    printf("[error] failed loading model\n");
    cleanUpModel(&model,&ctx,&smpl);
    return {};
  }
  printf("[log] completed loading model\n");
  printf("[log] loading context for model\n");
  ctx=llama_init_from_model(model, context_params);
  if(!ctx){
    printf("[log] failed loading conext for model\n");
    cleanUpModel(&model,&ctx,&smpl);
    return {};
  }
  printf("[log] completed loading context for model\n");
  printf("[log] loading sampler\n");
  smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  return (struct model_s){model,ctx,smpl};
}

#include <sys/stat.h>
#include <chrono>
int main(int argc,char* argv[]){
  auto start_time = chrono::high_resolution_clock::now();
  struct stat tmp={0};
  struct data_s data={0};
  struct model_s pipeline={0};
  bool backend_initialized=0;
  char*publ=0,*priv=0,**raw=0;
  size_t publ_size=0,priv_size=0;
  int raw_size=0,*rsize=0;
  if(stat(PUBLIC,&tmp)==0){
    publ=readFile(PUBLIC,&publ_size);
    if(publ)raw_size++;
    else printf("[error] failed to read %s\n",PUBLIC);
  }
  if(stat(PRIVATE,&tmp)==0){
    priv=readFile(PRIVATE,&priv_size);
    if(priv)raw_size++;
    else printf("[error] failed to read %s\n",PRIVATE);
  }
  if(!publ&&!priv){
    printf("[very very extreme fatal error] there is no input file ;-;\n");
    if(publ)free(publ);
    if(priv)free(priv);
    return 1;
  }
  raw=(char**)malloc(raw_size*sizeof(int*));//black magic
  rsize=(int*)alloca(raw_size*sizeof(int));
  int t=0;
  if(!raw||!rsize){
    printf("[log] failed allocating stuff at main()\n");
    if(publ)free(publ);
    if(priv)free(priv);
    return 1;
  }
  if(publ){
    rsize[t]=publ_size;
    raw[t++]=publ;
  }
  if(priv){
    rsize[t]=priv_size;
    raw[t++]=priv;
  }
  data=parsingDataJson(raw,rsize,raw_size);
  if(argc<2)pipeline=initializeModel(MODEL,&backend_initialized);
  else pipeline=initializeModel(TEST,&backend_initialized);
  if (!pipeline.ctx || !pipeline.model) {
    printf("[fatal error] failed initializing neural engine backend structures\n");
    cleanUpData(&data.data,&data.dictionary,&data.total_question,&data.total_char);
    cleanUpModel(&pipeline.model,&pipeline.ctx,&pipeline.smpl);
    if(backend_initialized)llama_backend_free();
    if(raw)free(raw);
    if(publ)free(publ);
    if(priv)free(priv);
    return 1;
  }
  if(data.total_question>0){
    FILE*out=fopen(OUTPUT,"w");
    if(!out){
      printf("[fatal error] failed creating output file: %s\n",OUTPUT);
      cleanUpData(&data.data,&data.dictionary,&data.total_question,&data.total_char);
      cleanUpModel(&pipeline.model,&pipeline.ctx,&pipeline.smpl);
      if(backend_initialized)llama_backend_free();
      if(raw)free(raw);
      if(publ)free(publ);
      if(priv)free(priv);
      return 1;
    }
    printf("[log] starting processing answer\n");
    fprintf(out,"qid,answer\n");
    struct token_s shared_token=initSharedPromptToken(&pipeline,data.total_option-2);
    int seq_pos=0;
    processRangeQuestion(&data,&pipeline,&shared_token,0,data.total_question-1,&seq_pos,out);
    auto end_time=chrono::high_resolution_clock::now();
    double total_seconds=chrono::duration<double>(end_time-start_time).count();
    printf("[log] completed processed %d questions in %.2f seconds (Avg: %.3fs per question).\n",data.total_question,total_seconds,total_seconds/data.total_question);
    cleanUpToken(&shared_token.data,&shared_token.dictionary,0,0);
    fclose(out);
  }
end:
  cleanUpData(&data.data,&data.dictionary,&data.total_question,&data.total_char);
  cleanUpModel(&pipeline.model,&pipeline.ctx,&pipeline.smpl);
  if(backend_initialized)llama_backend_free();
  if(raw)free(raw);
  if(publ)free(publ);
  if(priv)free(priv);
  return 0;
}
