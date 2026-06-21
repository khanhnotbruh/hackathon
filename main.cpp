/*
 * -- random comment --
 * "im going to be autistic" -17/6/2026
 * "lý do tại s e ko có log cho parsing hay là read file
 * vì nó chạy quá nhanh :))"-18/6/2026-
 * "pls check line 210"-19/6/2026-
 * "dcm s t ko đặt tên cho cái dictionary là pos ;-;" -20/6/2026-
 * "i hate goto"-21/6/2026-
 * -- road map --
 *  parses->tokenize->batch->decode->model (super simple)
 *  mất 3 ngày lang thang ở python và docker
 *  1d dành cho parse: debug 3d->4 ngày
 *  3d dành cho model: debug 3d->6 ngày
 * -- note--
 *  code này được viết 100% bởi e (i use arch btw)
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
#include "json.hpp"
#include "llama.h"
#include <alloca.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
using namespace std;
#define MODEL "./models/Qwen3.5-4B-Q4_K_M.gguf"
#define EMBEDDING "./models/Qwen3.5-9B-Q4_K_M.gguf" // dangerous 10gb file
#define PUBLIC "./data/public-data.json"
#define PRIVATE "./data/private-data.json"
#define OUTPUT "./output/pred.csv"

struct data_s{
  char*data;
  char***dictionary;
  int total_question,total_char;
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
  if(model&&*model){llama_free_model(*model);*model=0;}
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
// pass the pointer to the data's string
// pls dont pass more than two file into this
struct data_s parsingDataJson(char*raw[],int raw_size[],int size){
  if(!raw){
    printf("[error] passing null pointer to parsing??\n");
    return data_s{};
  }
  int arr_num=0,out_size=0,total_object=0,total_char=0;
  struct json_value_s**raw_root=(struct json_value_s**)alloca(size*sizeof(struct json_value_s*));
  for(int i=0;i<size;i++){
    raw_root[i]=json_parse(raw[i],raw_size[i]);
    if(!raw_root[i]||raw_root[i]->type!=json_type_array){
      printf("[error] failed to parse\n");
      return{};
    }
    total_object+=((struct json_array_s*)raw_root[i]->payload)->size;
    total_char+=raw_size[i];
  }
  char*data=(char*)malloc(total_char+1);//add only one '\0'
  char***dictionary=(char***)malloc(total_object*sizeof(char**));
  if(!data||!dictionary){
    printf("[error] allocation failed while parsing json file\n");
    cleanUpData(&data,&dictionary,0,0);
    return{};
  }
  char*write_ptr=data;
  for(int i=0;i<total_object;i++){
    dictionary[i]=(const char**)malloc(6*sizeof(const char*));//qid,question,a,b,c,d
    if(!dictionary[i]){
      printf("[error] allocation failed while parsing json file\n");
      cleanUpData(&data,&dictionary,0,0);
      return{};
    }
    for(int j=0;j<6;j++)dictionary[i][j]=0;
  }
  for(int i=0;i<size;i++){
    struct json_array_element_s *array_node=((struct json_array_s*)raw_root[i]->payload)->start;
    while(array_node){
      if(array_node->value->type!=json_type_object){
        printf("[error] unexpected value at object number %d\n",arr_num+1);
        cleanUpData(&data,&dictionary,0,0);
        return{};
      }
      struct json_object_element_s*object_node=((struct json_object_s*)array_node->value->payload)->start;
      int obj_num=0;
      while(object_node){
        obj_num++;//1-indexed
        switch (object_node->value->type){
          case json_type_string:{
            struct json_string_s*string_node=(struct json_string_s*)object_node->value->payload;
            if(object_node->name->string_size==8&&memcmp(object_node->name->string,"question",8)==0){
              dictionary[arr_num][1]=write_ptr;
              memcpy(write_ptr,string_node->string,string_node->string_size);
              write_ptr+=string_node->string_size;
            }else if(object_node->name->string_size==3&&memcmp(object_node->name->string,"qid",3)==0){
              dictionary[arr_num][0]=write_ptr;
              memcpy(write_ptr,string_node->string,string_node->string_size);
              write_ptr+=string_node->string_size;
            }else{
              printf("[error] unexpected naming at object number %d, pls use \"question\" or \"qid\"\n",obj_num);
              cleanUpData(&data,&dictionary,0,0);
              return{};
            }
            break;
          }
          case json_type_array:{
            if(object_node->name->string_size==7&&memcmp(object_node->name->string,"choices",7)==0){
              struct json_array_element_s*choice_node=((struct json_array_s*)object_node->value->payload)->start;
              int ans_num=0;
              while(choice_node){
                struct json_string_s*string_node=(struct json_string_s*)choice_node->value->payload;
                dictionary[arr_num][2+ans_num]=write_ptr;
                memcpy(write_ptr,string_node->string,string_node->string_size);
                write_ptr+=string_node->string_size;
                ans_num++;
                choice_node=choice_node->next;
              }
            }else{
              printf("[error] unexpected naming at object number %d, pls use \"choices\"\n",obj_num);
              cleanUpData(&data,&dictionary,0,0);
              return{};
            }
            break;
          }
          default:
            printf("[error] unexpected type at object number %d\n",obj_num);
            cleanUpData(&data,&dictionary,0,0);
            return{};
        }
        object_node=object_node->next;
      }
      array_node=array_node->next;
      arr_num++;
    }
  }
  *write_ptr='\0';
  out_size=write_ptr-data;
  return {data,dictionary,total_object,out_size};
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
  llama_token**dictionary=(llama_token**)malloc((r-l+1)*sizeof(token));
  if(!token||!dictionary){
    printf("[error] allocation failed while initializing space for tokenizing");
    cleanUpToken(&token,&dictionary,0,0);
    return {};
  }
  llama_token*write_ptr=token;
  for(int i=0;i<(r-l+1);i++)dictionary[i]=0;
  int used_size=0;
  for(int i=l;i<=r;i++){
    int cur_size=llama_tokenize(model,prompt_pos[i],len[i],write_ptr,max_token-used_size,false,true);
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
token_s initSharedPromptToken(struct model_s*model){
  const char*prompt[7]={
    "<|im_start|>system\nBạn là một trợ lý ảo vô cùng chính xác, chuyên xử lý bài thi trắc nghiệm. Nhiệm vụ của bạn là đọc câu hỏi và các lựa chọn được cung cấp, sau đó phân tích để tìm ra đáp án đúng nhất.\nYÊU CẦU BẮT BUỘC: Chỉ được phép trả về duy nhất một chữ cái đại diện cho đáp án đúng (A, B, C, hoặc D). Không giải thích, không thêm khoảng trắng, không lặp lại câu hỏi.<|im_end|>\n",
    "<|im_start|>user\n",
    "\nCác lựa chọn:\n A: ",
    "\n B: ",
    "\n C: ",
    "\n D: ",
    "\nĐáp án đúng là:<|im_end|>\n<|im_start|>assistant\n"
  };
  int strsize[7]={0};
  for(int i=0;i<7;i++)strsize[i]=strlen(prompt[i]);
  return tokenizeRange(model->model,prompt,strsize,7,0,6);
};
void processRangeQuestion(struct data_s*data,struct model_s*model,struct token_s*shared_token,int sqidx,int eqidx,FILE*out){
  if(sqidx>eqidx)return;
  llama_token*shared_pos[8]={
    shared_token->dictionary[0], shared_token->dictionary[1], shared_token->dictionary[2], shared_token->dictionary[3], 
    shared_token->dictionary[4], shared_token->dictionary[5], shared_token->dictionary[6], shared_token->dictionary[0]+shared_token->total_number
  };
  int total_question=eqidx-sqidx+1;
  int shared_size=shared_pos[1]-shared_pos[0];
  int segment_size=((eqidx+1==data->total_question)?data->dictionary[sqidx][0]+data->total_char:data->dictionary[eqidx+1][0])-data->dictionary[sqidx][0];//??
  int max_token=segment_size+total_question*(shared_pos[7]-shared_pos[1])+shared_size+(1<<10);//black magic
  int max_batch=llama_n_batch(model->ctx);
  if(sqidx==eqidx&&max_token>max_batch){
    int qid_size = data->dictionary[sqidx][1]-data->dictionary[sqidx][0];
    printf("[warning] encountered a very big question (%.*s:%d tokens) relative to your small ass vram",qid_size,data->dictionary[sqidx][0],max_token);
    fprintf(out, "%.*s,C\n",qid_size,data->dictionary[sqidx][0],data->dictionary[sqidx][0]);
    fflush(out);
    return;
  }
  //recursion stuff
  if(max_batch>=max_token){
    printf("[log] processing %d question",total_question);
    struct llama_batch batch=llama_batch_init(max_token,0,total_question);
    //creating shared prompt
    batch.n_token=0;
    int pos=0;
    appendToBatch(batch,shared_pos[0],&pos,shared_size,sqidx,eqidx);
    for(int bidx=0;bidx<total_question;bidx++){
      int qidx=bidx+sqidx;
      const char* prompt_pos[6] = {
        data->dictionary[qidx][1],data->dictionary[qidx][2],data->dictionary[qidx][3],data->dictionary[qidx][4], 
        data->dictionary[qidx][5],(qidx+1==data->total_question)?(data->data+data->total_char):data->dictionary[qidx+1][0]
      };
      int strsize[5]={0};
      for(int i=0;i<5;i++)strsize[i]=prompt_pos[i+1]-prompt_pos[i];
      struct token_s tk=tokenizeRange(model->model,prompt_pos,strsize,5,0,4);
      llama_token*tk_pos[6]={
        tk.dictionary[0],tk.dictionary[1],tk.dictionary[2],tk.dictionary[3], 
        tk.dictionary[4],tk.dictionary[0]+tk.total_number
      }
      for(int i=0;i<5;i++){
        int j=i+1;
        appendToBatch(batch,shared_pos[j],&pos,shared_pos[j+1]-shared_pos[j],bidx,bidx);
        appendToBatch(batch,tk_pos[i],&pos,tk_pos[j]-tk_pos[i],bidx,bidx);
      }
      appendToBatch(batch,shared_pos[6],&pos,shared_pos[7]-shared_pos[6],bidx,bidx);
      cleanUpToken(&tk.data,&tk.dictionary,&tk.total_token,&tk.total_number);
    }
    // decode
    if(!llama_decode(model->ctx,batch)){
      for(int bidx=0;bidx<total_question;bidx++){
        int qidx=sqidx+bidx;
        llama_token prediction=llama_sampler_sample(model->smpl,model->ctx,bidx);
        char out_buf[64];
        int res_len=llama_token_to_piece(model->model,prediction,out_buf,sizeof(out_buf),0,false);
        char ans='C';
        for (int i = 0; i < res_len; i++) {
          if (out_buf[i]>='A'&&out_buf[i]<='D'){ans=out_buf[i];break;}
          if (out_buf[i]>='a'&&out_buf[i]<='d'){ans=out_buf[i]&~32;break;}
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
    printf("[log] devided into two branch: from %d to %d and %d to %d",sqidx,mid,mid+1,eqidx);
    processRangeQuestion(data,model,shared_token,sqidx,mid,out);
    processRangeQuestion(data,model,shared_token,mid+1,eqidx,out);
  }
}

struct model_s initializeModel(char*model_path,bool*init){
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
  context_params.n_batch=4096;
  context_params.n_ctx=4096;
  context_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  printf("[log] completed loading parameters\n");
  printf("[log] loading model: %s\n",model_path);
  model=llama_load_model_from_file(model_path, model_params);
  if(!model){
    printf("[error] failed loading model");
    cleanUpModel(&model,&ctx,&smpl);
    return {};
  }
  printf("[log] completed loading model\n");
  printf("[log] loading context for model\n");
  ctx=llama_new_context_with_model(model,context_params);
  if(!ctx){
    printf("[log] failed loading conext for model\n");
    cleanUpModel(&model,&ctx,&smpl);
    return {};
  }
  printf("[log] completed loading context for model\n");
  printf("[log] loading sampler");
  smpl=llama_sampler_init_chain();
  llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
  return (struct model_s){model,ctx,smpl};
}

#include <sys/stat.h>
#include <chrono>
int main(){
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
    else printf("[error] failed to read %s",PUBLIC);
  }
  if(stat(PRIVATE,&tmp)==0){
    priv=readFile(PRIVATE,&priv_size);
    if(priv)raw_size++;
    else printf("[error] failed to read %s",PRIVATE);
  }
  if(!publ&&!priv){
    printf("[very very extreme fatal error] there is no input file ;-;");
    if(publ)free(publ);
    if(priv)free(priv);
    return 1;
  }
  raw=(char**)malloc(raw_size*sizeof(int*));//black magic
  rsize=(int*)alloca(raw_size*sizeof(int));
  int t=0;
  if(!raw||!rsize){
    printf("[log] failed allocating stuff at main()");
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
  pipeline=initializeModel(MODEL,&backend_initialized);
  if (!pipeline.ctx || !pipeline.model) {
    printf("[fatal error] failed initializing neural engine backend structures.\n");
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
    printf("[log] starting processing answer");
    fprintf(out,"qid,answer\n");
    struct token_s shared_token=initSharedPromptToken(pipeline);
    processRangeQuestion(&data,&pipeline,&shared_token,0,data.total_question-1,out);
    auto end_time=chrono::high_resolution_clock::now();
    double total_seconds=chrono::duration<double>(end_time-start_time).count();
    printf("[log] completed processed %d questions in %.2f seconds (Avg: %.3fs per question).\n",data.total_question,total_seconds,total_seconds/data.total_question);
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
