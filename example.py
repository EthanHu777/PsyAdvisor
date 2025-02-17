from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch
import opencc

model_name = "Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_model_path = "./output/lora_model"
lora_model = AutoModelForCausalLM.from_pretrained(lora_model_path)

lora_config = LoraConfig.from_pretrained(lora_model_path)
model = get_peft_model(model, lora_config)
model.load_state_dict(lora_model.state_dict(), strict=False)

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if torch.cuda.is_available():
    model.half()
torch.backends.cudnn.benchmark = True

cc = opencc.OpenCC('t2s.json')

soulchat_model_path = './SoulChat'
soulchat_tokenizer = AutoTokenizer.from_pretrained(soulchat_model_path)
soulchat_model = AutoModelForCausalLM.from_pretrained(soulchat_model_path)
soulchat_model.to(device)

def generate_response(input_text):
    prompt = f'''你是一个心理咨询策略规划助手，你的任务是根据来访者的输入，为心理咨询师提供一种策略建议。
    请你遵循如下要求进行输出：\n
    1. 策略名称记忆：请你学习目前已知所有12种可能的策略：\n
    - "重述"\n
    - "最小的鼓励"\n
    - "情感反映"\n
    - "认可和安慰"\n
    - "回答"\n
    - "面质"\n
    - "自我暴露"\n
    - "解释"\n
    - "探询主观信息"\n
    - "探询客观信息"\n
    - "邀请探索（或采取）新行动"\n
    - "邀请采用新视角"\n
    2. 选择"策略"：根据来访者的输入{input_text}，从上述12种策略中匹配出最合适的一种。\n
    3. 解释：告诉咨询师你选择这一策略的原因，例如："我使用这一策略是为了询问来访者的内心想法、感受、动机等"。\n
    4. 举例：举一个简短的例子告诉咨询师如何回复来访者，注意用友好、亲切的语气，例如："可以结合你生活中的例子和我说说吗？"来询问来访者。\n
    5. 使用以下格式输出：["策略"]\n 解释\n 举例。\n
    6. 以上格式是告诉咨询师如何给来访者回复，而不是直接给来访者回复，注意语义连贯，不能出现错别字。\n
    请注意，你推理的中间过程和prompt都不要输出，至少提供一种已有的策略名称，只能使用自然语言回复，不能有其他格式。
    此外，你需要明白你是在给咨询师提供建议，而不是直接回答来访者的问题，只能使用中文回答，总字数100字内。\n
    '''
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    attention_mask = (inputs['input_ids'] != tokenizer.pad_token_id).long()
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|endoftext|>']})
    model.resize_token_embeddings(len(tokenizer))
    eos_token = tokenizer.encode("<|endoftext|>")[0]
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=attention_mask, 
        max_new_tokens=256, 
        num_beams=1,
        top_p=0.2,
        top_k=5,
        no_repeat_ngram_size=2,
        eos_token_id=eos_token,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    start_index = response.find("[", response.find("[") + 1)
    if start_index != -1:
        response = response[start_index:]
        
    first_period_index = response.find("。", response.find("。")+1)
    if first_period_index != -1:
        response = response[:first_period_index + 1]
    
    prompt_str = prompt.strip()
    if response.startswith(prompt_str):
        response = response[len(prompt_str):].strip()
    
    response = cc.convert(response)
    
    return response

def chat_with_model():
    while True:
        user_input = input("来访者：")
        if user_input.lower() in ["退出", "1"]:
            print("再见！")
            break
        
        response = generate_response(user_input)
        print(f"策略建议：{response}")
        
        combined_input = f"来访者提问：{user_input}\n你作为一个专业的心理咨询师，请根据以下策略建议，为来访者给出一个适当的回复：\n策略建议：{response}\n"
        
        soulchat_inputs = soulchat_tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        soulchat_attention_mask = (soulchat_inputs['input_ids'] != soulchat_tokenizer.pad_token_id).long()
        soulchat_outputs = soulchat_model.generate(
            soulchat_inputs['input_ids'], 
            attention_mask=soulchat_attention_mask, 
            max_new_tokens=256, 
            num_beams=2,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=2,
            temperature=0.9
        )
        
        soulchat_response = soulchat_tokenizer.decode(soulchat_outputs[0], skip_special_tokens=True)
        
        soulchat_response = soulchat_response.replace(combined_input, "").strip()
        
        print(f"SoulChat：{soulchat_response}")

chat_with_model()
