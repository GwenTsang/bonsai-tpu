// Reference-only CPU oracle using the pinned Prism llama.cpp public API.
#include "llama.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct Capture {
    std::string directory;
    int step=0;
    int serial=0;
    std::ofstream manifest;
};

static bool capture(ggml_tensor *t, bool ask, void *opaque) {
    auto &c=*static_cast<Capture*>(opaque);
    std::string name=t->name;
    bool selected=c.step<2 && (name=="model.input_embed" || name=="embd" ||
        (name.size()>2 && (name.substr(name.size()-2)=="-0" || name.substr(name.size()-2)=="-3")));
    if(ask) return selected;
    if(!selected) return true;
    for(char &v:name) if(!std::isalnum(static_cast<unsigned char>(v)) && v!='-' && v!='_') v='_';
    const auto filename=std::to_string(c.step)+"_"+std::to_string(c.serial++)+"_"+name+".bin";
    std::vector<char> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t,data.data(),0,data.size());
    std::ofstream f(c.directory+"/"+filename,std::ios::binary);
    f.write(data.data(),data.size());
    c.manifest<<"{\"file\":\""<<filename<<"\",\"name\":\""<<name<<"\",\"step\":"<<c.step<<",\"type\":"<<t->type<<",\"shape\":[";
    for(int i=0;i<4;++i)c.manifest<<(i?",":"")<<t->ne[i];
    c.manifest<<"],\"strides\":[";
    for(int i=0;i<4;++i)c.manifest<<(i?",":"")<<t->nb[i];
    c.manifest<<"]}\n";
    return f.good() && c.manifest.good();
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr,"usage: dump_logits model prompt out_prefix steps\n"); return 2; }
    const std::string prompt=argv[2], out=argv[3];
    const int steps=std::stoi(argv[4]);
    if (steps<1) return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    auto *model=llama_model_load_from_file(argv[1],mp);
    if (!model) return 3;
    auto *vocab=llama_model_get_vocab(model);
    int count=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);
    if (count<1) return 4;
    std::vector<llama_token> tokens(count);
    if(llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),count,true,true)!=count) return 5;
    auto cp=llama_context_default_params();
    cp.n_ctx=std::max(512,count+steps);cp.n_batch=std::max(count,32);cp.n_ubatch=cp.n_batch;
    cp.n_threads=8;cp.n_threads_batch=8;cp.no_perf=false;
    Capture captures;
    if(const char *path=std::getenv("BONSAI_CAPTURE_DIR")) {
        captures.directory=path;std::filesystem::create_directories(path);
        captures.manifest.open(captures.directory+"/manifest.jsonl");
        cp.cb_eval=capture;cp.cb_eval_user_data=&captures;
    }
    auto *ctx=llama_init_from_model(model,cp);
    if(!ctx) return 6;
    auto *batch=llama_batch_ext_init(ctx);
    const int nv=llama_vocab_n_tokens(vocab);
    std::ofstream logits(out+".f32",std::ios::binary),meta(out+".json");
    meta<<"{\"vocab_size\":"<<nv<<",\"steps\":"<<steps<<",\"prompt_ids\":[";
    for(int i=0;i<count;++i) meta<<(i?",":"")<<tokens[i];
    meta<<"],\"greedy_ids\":[";
    int pos=0;
    for(int step=0;step<steps;++step) {
        captures.step=step;
        llama_batch_ext_clear(batch);
        for(size_t i=0;i<tokens.size();++i) {
            int idx=llama_batch_ext_add_token(batch,0,tokens[i]);
            llama_pos p=pos+i;llama_batch_ext_set_pos(batch,idx,&p);
        }
        llama_batch_ext_set_output_logits(batch,tokens.size()-1,true);
        if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch)) return 7;
        pos+=tokens.size();
        const float *values=llama_get_logits_ith(ctx,-1);
        if(!values) return 8;
        logits.write(reinterpret_cast<const char*>(values),nv*sizeof(float));
        int id=std::max_element(values,values+nv)-values;
        meta<<(step?",":"")<<id;
        tokens.assign(1,id);
    }
    meta<<"]}\n";
    logits.close();meta.close();
    llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
    return logits.good()&&meta.good()?0:9;
}
