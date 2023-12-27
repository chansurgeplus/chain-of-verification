import argparse
from dotenv import load_dotenv
from pprint import pprint

from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

from route_chain import RouteCOVEChain
from hf_langchain import ChatHuggingFace

load_dotenv("/workspace/.env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Chain of Verification (CoVE) parser.')
    parser.add_argument('--question',  
                        type = str,
                        required = True,
                        help ='The original question user wants to ask')
    parser.add_argument('--llm-name',  
                        type = str,
                        required = False,
                        default = "mistralai/Mistral-7B-Instruct-v0.2",
                        help ='The huggingface model id')
    parser.add_argument('--temperature',  
                        type = float,
                        required = False,
                        default = 0.1,
                        help ='The temperature of the llm')
    parser.add_argument('--max-tokens',  
                        type = int,
                        required = False,
                        default = 4096,
                        help ='The max_tokens of the llm')
    parser.add_argument('--show-intermediate-steps',  
                        type = bool,
                        required = False,
                        default = True,
                        help ='The max_tokens of the llm')
    args = parser.parse_args()
    
    original_query = args.question

    chain_llm = ChatHuggingFace(model_id=args.llm_name, model_kwargs={
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    )
    route_llm = ChatHuggingFace(model_id=args.llm_name, model_kwargs={
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    )
    
    router_cove_chain_instance = RouteCOVEChain(original_query, route_llm, chain_llm, args.show_intermediate_steps)
    router_cove_chain = router_cove_chain_instance()
    router_cove_chain_result = router_cove_chain({"original_question":original_query})
    
    if args.show_intermediate_steps:
        print("\n" + 80*"#" + "\n")
        pprint(router_cove_chain_result)
        print("\n" + 80*"#" + "\n")
    print("Final Answer: {}".format(router_cove_chain_result["final_answer"]))
    