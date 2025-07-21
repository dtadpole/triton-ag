from transformers import AutoTokenizer

def proper_qwen_tokenization(tokenizer, text):
    """正确查看 Qwen 分词结果"""
    
    print(f"原文: '{text}'")
    
    # 1. 获取 token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Token IDs: {token_ids}")
    
    # 2. 获取 tokens (可能显示为乱码)
    tokens = tokenizer.tokenize(text)
    print(f"Raw tokens: {tokens}")
    
    # 3. 正确方法：逐个解码每个 token ID
    print("正确的 token 解码:")
    for i, (token_id, token) in enumerate(zip(token_ids, tokens)):
        # 解码单个 token ID
        decoded_token = tokenizer.decode([token_id])
        print(f"  {i}: ID={token_id}, Raw='{token}', Decoded='{decoded_token}'")
    
    # 4. 完整重建
    reconstructed = tokenizer.decode(token_ids)
    print(f"重建文本: '{reconstructed}'")
    print(f"匹配原文: {reconstructed == text}")

# 测试
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
proper_qwen_tokenization(tokenizer, "你好世界")