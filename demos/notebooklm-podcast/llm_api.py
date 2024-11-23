# -*- coding: utf-8 -*-
# @Time    : 2024/11/3
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : ChatTTSPlus
# @FileName: llm_api.py

import os
import json
import copy
import base64
import pdb

from openai import OpenAI


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def init_script_writer_prompt(speaker1_gender="woman", speaker2_gender="man"):
    operation_history = []
    sysetm_prompt = f"""
    You are a renowned podcast scriptwriter, having worked as a ghostwriter for personalities like Joe Rogan, Lex Fridman, Ben Shapiro, and Tim Ferris. Your task is to create a dialogue based on a provided text by user, 
    incorporating interjections such as "umm," "hmmm," and "right" from the second speaker. The conversation should be highly engaging; occasional digressions are acceptable, 
    but the discussion should primarily revolve around the main topic.
    
    The dialogue involves two speakers:
    
    Speaker 1: This person, a {speaker1_gender}, leads the conversation, teaching Speaker 2 and providing vivid anecdotes and analogies during explanations. They are a charismatic teacher known for their compelling storytelling.
    
    Speaker 2: This person, a {speaker2_gender}, keeps the conversation flowing with follow-up questions, exhibiting high levels of excitement or confusion. Their inquisitive nature leads them to ask interesting confirmation questions.
    
    Ensure that any tangents introduced by Speaker 2 are intriguing or entertaining. Interruptions during explanations and interjections like "hmm" and "umm" from Speaker 2 should be present throughout the dialogue.
    
    Your output should resemble a genuine podcast, with every minute nuance documented in as much detail as possible. Begin with a captivating overview to welcome the listeners, keeping it intriguing and bordering on clickbait.
    
    Always commence your response with Speaker 1. Do not provide separate episode titles; instead, let Speaker 1 incorporate it into their dialogue. There should be no chapter titles; strictly provide the dialogues.
    """
    operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return operation_history


def init_script_rewriter_prompt():
    operation_history = []
    system_prompt = """
    As an Oscar-winning international screenwriter, you've collaborated with many award-winning podcasters. Your current task is to rewrite the following podcast transcript for an AI Text-To-Speech Pipeline. A rudimentary AI initially wrote this transcript, and it now requires your expertise to make it engaging.

    Two different voice engines will simulate Speaker 1 and Speaker 2. 

    Speaker 1: This character leads the conversation and educates Speaker 2. They are known for providing captivating teachings, enriched with compelling anecdotes and analogies.

    Speaker 2: This character maintains the conversation's flow by asking follow-up questions. They exhibit high levels of excitement or confusion and have a curious mindset, often asking intriguing confirmation questions.

    Speaker 2's tangents should be wild or interesting. Interruptions during explanations and interjections like "hmm" and "umm" from Speaker 2 should be present throughout the dialogue.

    Note: The Text-To-Speech engine for Speaker 1 struggles with "umms" and "hmms," so maintain a straight text for this speaker. For Speaker 2, feel free to use "umm," "hmm," [sigh], and [laughs] to convey expressions.

    The output should resemble a genuine podcast, with every minute detail documented meticulously. Begin with a captivating overview to welcome the listeners, keeping it intriguing and bordering on clickbait.

    Your response should start directly with Speaker 1 and strictly be returned as a list of tuples. No additional text should be included outside of the list.

    Example of response:
    [
        ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
        ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
        ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
        ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
    ]
    """
    operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
    return operation_history


def inference_openai_chat(messages, model, api_url, token, max_tokens=2048, temperature=0.4, seed=1234):
    client = OpenAI(
        base_url=api_url,
        api_key=token,
    )

    data = {
        "model": model,
        "messages": [],
        "max_tokens": max_tokens,
        'temperature': temperature,
        "seed": seed
    }

    for role, content in messages:
        data["messages"].append({"role": role, "content": content})

    completion = client.chat.completions.create(
        **data
    )
    return completion.choices[0].message.content


def add_response(role, prompt, chat_history, image=None):
    new_chat_history = copy.deepcopy(chat_history)
    if image:
        base64_image = encode_image(image)
        content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
    else:
        content = [
            {
                "type": "text",
                "text": prompt
            },
        ]
    new_chat_history.append([role, content])
    return new_chat_history


if __name__ == '__main__':
    import utils
    import pickle

    base_url = "https://api2.aigcbest.top/v1"
    api_token = "sk-2gnUzC8dXpiFBMgBaPEJlS8WasXdNSQx2JKi3ZOdCSG2h4G0"
    gpt_model = "gpt-4o-mini"
    pdf_txt_file = "../../data/pdfs/AnimateAnyone.txt"
    script_pkl = os.path.splitext(pdf_txt_file)[0] + "-script.pkl"
    re_script_pkl = os.path.splitext(pdf_txt_file)[0] + "-script-rewrite.pkl"

    chat_writer = init_script_writer_prompt()
    pdf_texts = utils.read_file_to_string(pdf_txt_file)
    chat_writer = add_response("user", pdf_texts, chat_writer)
    output_writer_texts = []
    output_writer = inference_openai_chat(chat_writer, gpt_model, base_url, api_token, max_tokens=8192)
    print(output_writer)
    chat_writer = add_response("assistant", output_writer, chat_writer)

    with open(script_pkl, 'wb') as file:
        pickle.dump(output_writer, file)

    with open(script_pkl, 'rb') as file:
        script_texts = pickle.load(file)
    chat_rewriter = init_script_rewriter_prompt()
    chat_rewriter = add_response("user", script_texts, chat_rewriter)
    output_rewriter = inference_openai_chat(chat_rewriter, gpt_model, base_url, api_token, max_tokens=8192)
    print(output_rewriter)

    with open(re_script_pkl, 'wb') as file:
        pickle.dump(output_rewriter, file)
