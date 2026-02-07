from transformers import AutoTokenizer

def compute_pixel_token_ids(tokenizer):
    '''
    返回一个 (256,) 大小的数组: 0-255 token_id in LLM vocabunary
    '''
    pixel_value = list(range(256))
    pixel_token_ids = [tokenizer.encode(str(value))[0] for value in pixel_value]
    return pixel_token_ids

def compute_token_ids_to_pixel(tokenizer):
    """
    返回一个 dict: { token_id: pixel_value }。
    """
    pixel_token_ids = compute_pixel_token_ids(tokenizer)
    tokenid_to_pixel = {}
    for pixel_val, tid in enumerate(pixel_token_ids):
        tokenid_to_pixel[tid] = pixel_val
    
    return tokenid_to_pixel

def tokenid_to_pixel(token_id, tokenizer):
    """
    查询单个 token_id 对应的 pixel 值。
    """
    mapping = compute_token_ids_to_pixel(tokenizer)
    pixelvalue = mapping.get(token_id, None)
    
    return pixelvalue


if __name__ == "__main__":
    model_name = "../Model/diffugpt-s"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pixel_token_ids = compute_pixel_token_ids(tokenizer)
    print(f"pixel_token_ids: {pixel_token_ids}")
    token_id_to_pixel = compute_token_ids_to_pixel(tokenizer)
    print(f"token_id_to_pixel: {token_id_to_pixel}")