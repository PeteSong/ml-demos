import random

import torch


def predict_masked_token_with_bert():
    def load_tokenizer():
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        print(f'''
    type: {type(tokenizer)}
    bert tokenizer: {tokenizer}

    cls/sep/mask token: {tokenizer.cls_token}, {tokenizer.sep_token}, {tokenizer.mask_token}
    cls/sep/mask token id: {tokenizer.cls_token_id}, {tokenizer.sep_token_id}, {tokenizer.mask_token_id}
        ''')
        return tokenizer

    def load_model():
        masked_lm_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForMaskedLM', 'bert-base-cased')
        print(f'''
    type: {type(masked_lm_model)}
    model: {masked_lm_model}
        ''')
        return masked_lm_model

    def get_tokens(tokenizer, *args):
        indexed_tokens = tokenizer.encode(*args, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(map(str, indexed_tokens))
        tokens_text = tokenizer.decode(indexed_tokens)
        print(f'''
type: {type(indexed_tokens)}    
indexed tokens: {indexed_tokens}
size: {len(indexed_tokens)}

tokens: {tokens}
size: {len(tokens)}
text: {tokens_text}

        ''')
        return indexed_tokens

    def predict(mdl, tokenizer, indexed_tokens, token_type_ids, mask_token_id):
        idxs = range(0, len(indexed_tokens))
        masked_idx = random.choice(idxs)
        masked_tokens = indexed_tokens.copy()
        masked_tokens[masked_idx] = mask_token_id
        print(f'''
original tokens ids: {indexed_tokens}
original tokens: {tokenizer.decode(indexed_tokens)}

masked tokens id: {masked_tokens}
masked tokens: {tokenizer.decode(masked_tokens)}

        ''')

        masked_token_tensor = torch.tensor([masked_tokens])
        with torch.no_grad():
            predictions = mdl(masked_token_tensor, token_type_ids=token_type_ids)
        predicted_indexed_token = torch.argmax(predictions[0][0], dim=1)[masked_idx].item()
        o_i = indexed_tokens[masked_idx]
        o_t = tokenizer.convert_ids_to_tokens([o_i])[0]
        p_i = predicted_indexed_token
        p_t = tokenizer.convert_ids_to_tokens([p_i])[0]
        print(f'''
masked indexed token: {o_i}  text: {o_t}
predicted indexed token: {p_i}  text: {p_t}    
is passed: {o_i == p_i}

        ''')

    masked_lm_model = load_model()
    tokenizer = load_tokenizer()
    text_1 = "I understand equations, both the simple and quadratical."
    text_2 = "What kind of equations do I understand?"
    indexed_tokens = get_tokens(tokenizer, text_1, text_2)
    segment_ids_tensor, indexed_tokens_tensor = get_segment_ids(indexed_tokens, tokenizer.sep_token_id)
    print(f'''
{indexed_tokens_tensor}
{segment_ids_tensor}
    ''')

    predict(masked_lm_model, tokenizer, indexed_tokens, segment_ids_tensor, tokenizer.mask_token_id)


def get_segment_ids(indexed_tokens, sep_token_id):
    segment_ids = []
    segment_id = 0
    for token in indexed_tokens:
        if token == sep_token_id:
            segment_id += 1
        segment_ids.append(segment_id)
    segment_ids[-1] -= 1  # ignore last [SEP]
    return torch.tensor([segment_ids]), torch.tensor([indexed_tokens])


predict_masked_token_with_bert()


def answer_question_with_bert():
    model = torch.hub.load('huggingface/pytorch-transformers',
                           'modelForQuestionAnswering',
                           'bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = torch.hub.load('huggingface/pytorch-transformers',
                               'tokenizer',
                               'bert-large-uncased-whole-word-masking-finetuned-squad')
    text_1 = "I understand equations, both the simple and quadratical."
    text_2 = "What kind of equations do I understand?"
    indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
    segment_ids_tensor, indexed_tokens_tensor = get_segment_ids(indexed_tokens, tokenizer.sep_token_id)
    with torch.no_grad():
        output = model(indexed_tokens_tensor, token_type_ids=segment_ids_tensor)
    s = torch.argmax(output.start_logits)
    e = torch.argmax(output.end_logits)
    answer_sequence = indexed_tokens[s:e + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_sequence)
    answer_text = tokenizer.decode(answer_sequence)
    print(f'''
answer indexes: {answer_sequence}
answer tokens: {answer_tokens}
answer: {answer_text}    
    ''')


answer_question_with_bert()
