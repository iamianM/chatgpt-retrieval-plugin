from typing import List
import openai, json
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from tenacity import retry, wait_random_exponential, stop_after_attempt, after, stop_after_delay

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

SECRETS = json.load(open("./OPENAI_KEY.json"))
openai.api_key = SECRETS["OPENAI_KEY"]

ATTEMPTS = 5

import logging
import logging.config

logging.config.fileConfig('logging_config.ini')
def log_retry(retry_state):
    logging.info(f'Retrying {retry_state.fn} after {retry_state.attempt_number} attempt(s) due to {retry_state.outcome}')

class SPLADE:
    def __init__(self, model):
        # check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        # move to gpu if available
        self.model.to(self.device)

    def __call__(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        inter = torch.log1p(torch.relu(logits[0]))
        token_max = torch.max(inter, dim=0)  # sum over input tokens
        nz_tokens = torch.where(token_max.values > 0)[0]
        nz_weights = token_max.values[nz_tokens]

        order = torch.sort(nz_weights, descending=True)
        nz_weights = nz_weights[order[1]]
        nz_tokens = nz_tokens[order[1]]
        return {
            'indices': nz_tokens.cpu().numpy().tolist(),
            'values': nz_weights.cpu().numpy().tolist()
        }

# splade = SPLADE("naver/splade-cocondenser-ensembledistil")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # Call the OpenAI API to get the embeddings
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")

    # Extract the embedding data from the response
    data = response["data"]  # type: ignore

    # Return the embeddings as a list of lists of floats
    return [result["embedding"] for result in data]

# Get dense embeddings using a pre-trained Sentence Transformer model
def get_embeddings(transcripts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(transcripts)
    return [e.tolist() for e in embeddings]

# Get sparse embeddings using TfidfVectorizer
def get_sparse_embeddings(transcripts):
    return []
    # return [splade(transcript) for transcript in transcripts]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
):
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # call the OpenAI chat completion API with the given messages
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    choices = response["choices"]  # type: ignore
    completion = choices[0].message.content.strip()
    print(f"Completion: {completion}")
    return completion

@retry(wait=wait_random_exponential(min=5, max=20), stop=(stop_after_attempt(ATTEMPTS) | stop_after_delay(120)), after=log_retry)
async def openai_response(model, messages, max_tokens, temperature):
    result = openai.ChatCompletion.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    ).choices[0].message['content']
    return result

from typing import List, Dict
import tiktoken

tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)
def get_messages2(question: str, results: List[Dict], token_len_max: int) -> str:
    token_len = 100000
    drop = 0
    
    while token_len > token_len_max:
        files_string = ""
        files_string += f"Files:\n"
        for i in range(len(results)):
            if i >= len(results) - drop:
                break
            
            filename = f"https://steno.ai/{results[i]['name'].lower().replace(' ', '-')}/{results[i]['slug']}"
            file_text = results[i]['text']
            file_string = f"###\n\"{filename}\"\n{file_text}\n"
            files_string += file_string
        files_string += "\n"
        messages = [{
            "role": "system",
            "content": f"{files_string}" \
            f"Above are segments of podcast transcripts relating to the users query. Use them to help you respond to the users request. " \
            f"Keep in mind the user cannot see the above. "
            f"When referencing an episode from the above use markdown format to make a clickable link. \n\n" \
            f"Question: {question}\n\n" \
            f"Answer:"
        }]

        token_len = len(tokenizer.encode(messages[0]['content'], disallowed_special=()))
        print(token_len)
        drop += 1
        
    print(messages)
    return results, messages