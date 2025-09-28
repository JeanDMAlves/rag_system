import evaluate
import pandas as pd

def evaluate_with_bertscore(results, ground_truth, lang="pt"):
    """
    Calcula BERTScore entre as respostas geradas pela RAG e as respostas esperadas.

    Args:
        results (list[dict]): lista com os resultados do sistema RAG, 
                              cada item precisa ter {"pergunta": ..., "resposta": ...}.
        ground_truth (list[dict]): lista com o ground truth, 
                                   cada item precisa ter {"question": ..., "relevant_chunks": [...]}
        lang (str): idioma, default "pt".

    Returns:
        pd.DataFrame: DataFrame com pergunta, resposta, ground truth e métricas BERTScore.
    """
    bertscore = evaluate.load("bertscore")

    predictions = [r["resposta"] for r in results]
    references = [" ".join(gt["relevant_chunks"]) for gt in ground_truth]

    scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang=lang
    )

    # Montar tabela final
    data = []
    for r, gt, p, prec, rec, f1 in zip(results, ground_truth, predictions,
                                       scores["precision"], scores["recall"], scores["f1"]):
        data.append({
            "pergunta": r["pergunta"],
            "resposta_rag": p,
            "ground_truth": " ".join(gt["relevant_chunks"]),
            "bertscore_precision": prec,
            "bertscore_recall": rec,
            "bertscore_f1": f1
        })

    df = pd.DataFrame(data)
    return df
