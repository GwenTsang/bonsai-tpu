// Reference-only tokenizer oracle (vocab-only load). Input: file of JSON-escaped-free lines; "\\n" literal -> newline.
#include "llama.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
int main(int argc,char**argv){
    if(argc!=3) return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();mp.vocab_only=true;
    auto *model=llama_model_load_from_file(argv[1],mp);
    if(!model) return 3;
    auto *vocab=llama_model_get_vocab(model);
    std::ifstream in(argv[2]);std::string line;
    printf("[");bool first=true;
    while(std::getline(in,line)){
        std::string s;
        for(size_t i=0;i<line.size();++i){ if(line[i]=='\\'&&i+1<line.size()&&line[i+1]=='n'){s+='\n';++i;} else s+=line[i]; }
        int count=-llama_tokenize(vocab,s.data(),s.size(),nullptr,0,false,true);
        std::vector<llama_token> t(count>0?count:0);
        if(count>0) llama_tokenize(vocab,s.data(),s.size(),t.data(),count,false,true);
        printf("%s[",first?"":",\n");first=false;
        for(int i=0;i<count;++i) printf("%s%d",i?",":"",t[i]);
        printf("]");
    }
    printf("]\n");
    llama_model_free(model);llama_backend_free();return 0;
}
