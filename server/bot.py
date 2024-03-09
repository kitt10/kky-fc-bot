from transformers import AutoModel, AutoTokenizer
import json
import numpy as np
import torch
"""
This code was created on hackhaton and use czech for dict key am sorry.
"""
class ChatModel:

    def __init__(self, **config):
        self.__dict__.update(**config)

        model_name = "Seznam/retromae-small-cs"
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        f = open("final_data.json")

        data_with_classes = json.load(f)
        self.data_with_classes = json.loads(data_with_classes)
        f.close()

    def _find_best_class_answer(self,model, tokenizer, user_question, class_questions_answers):
        question_tokens = tokenizer(user_question, return_tensors='pt', padding=True, truncation=True, max_length=512)
        question_embedding = model(**question_tokens).last_hidden_state[:, 0]

        best_score = -np.inf
        best_answer = None
        for item in class_questions_answers:
            sim = []
            for i in item["otázky"]:
                questions_tokens = tokenizer(i, return_tensors='pt', padding=True, truncation=True, max_length=512)
                questions_embeddings = model(**questions_tokens).last_hidden_state[:, 0]
                similarity = torch.nn.functional.cosine_similarity(question_embedding, questions_embeddings, dim=1)
                sim.append(similarity.detach().numpy())
            similarity = np.mean(np.array(sim))
            if similarity.item() > best_score:
                best_score = similarity.item()
                best_answer = item["odpověď"]

        return best_answer

    def turn(self, prompt):
        reaction = self._find_best_class_answer(self.model,self.tokenizer,prompt,self.data_with_classes)
        return reaction


if __name__ == "__main__":
    agent = ChatModel()
    print(agent.turn("Padání vlasů"))