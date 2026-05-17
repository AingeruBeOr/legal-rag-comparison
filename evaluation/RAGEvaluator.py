import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts.llm_as_a_judge import prompt as judge_prompt
from evaluation.llm_judge_inferencer import LLMJudgeInferencer
import evaluate as hf_evaluate

class RAGEvaluator:
    def __init__(self):
        self.llm_judge_inferencer = LLMJudgeInferencer()
        # Inicializar las métricas de HuggingFace una sola vez para no descargar/cargar en cada iteración
        self.bleu_metric = hf_evaluate.load("bleu")
        self.rouge_metric = hf_evaluate.load("rouge")
        self.bertscore_metric = hf_evaluate.load("bertscore")

    def llm_as_a_judge(self, result):
        question: str = result["question"]
        generated_answer: str = result["generated_answer"]
        retrieved_context: list[dict] = result["retrieved_context"]

        eval_prompt = judge_prompt.format(
            query=question,
            context="\n\n".join([f"Context {i+1} {rc['payload']['text']}" for i, rc in enumerate(retrieved_context)]),
            answer=generated_answer
        )
        eval_response = self.llm_judge_inferencer.judge(eval_prompt)
        return eval_response["context_relevance"], eval_response["groundedness"], eval_response["answer_relevance"]

    def retrieval_metrics(self, result):
        target_source = result.get("metadata").get("source")
        retrieved_context = result.get("retrieved_context", [])
        
        hit_rate = 0
        mrr = 0.0
        
        for i, context in enumerate(retrieved_context):
            doc_source = context.get("payload").get("source_file")
            
            # Comparamos si el nombre del archivo esperado está en el recuperado o viceversa
            if target_source in doc_source or doc_source in target_source:
                hit_rate = 1
                mrr = 1.0 / (i + 1)
                break

        return hit_rate, mrr

    def generation_metrics(self, generated_answer, target_answer):
        # BLEU (lexical) - huggingface evaluate
        try:
            bleu_res = self.bleu_metric.compute(predictions=[generated_answer], references=[target_answer])
            bleu = bleu_res.get('bleu', 0.0)
        except Exception:
            bleu = 0.0

        # ROUGE (lexical) - huggingface evaluate
        try:
            rouge_res = self.rouge_metric.compute(predictions=[generated_answer], references=[target_answer])
            rouge_l = rouge_res.get('rougeL', 0.0)
        except Exception:
            rouge_l = 0.0

        # BERTScore (semantic) - huggingface evaluate
        try:
            bertscore_res = self.bertscore_metric.compute(predictions=[generated_answer], references=[target_answer], lang="es", verbose=False)
            bertscore_f1 = bertscore_res.get('f1', [0.0])[0]
        except Exception:
            bertscore_f1 = 0.0

        return bleu, rouge_l, bertscore_f1

    def evaluate(self, result: dict) -> dict:
        context_relevance, groundedness, answer_relevance = self.llm_as_a_judge(result)

        hit_rate, mrr = self.retrieval_metrics(result)
        
        bleu, rouge_l, bertscore_f1 = self.generation_metrics(
            generated_answer=result.get("generated_answer"),
            target_answer=result.get("target")
        )

        return {
            "context_relevance": context_relevance,
            "groundedness": groundedness,
            "answer_relevance": answer_relevance,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "bleu": bleu,
            "rouge_l": rouge_l,
            "bertscore_f1": bertscore_f1
        }
