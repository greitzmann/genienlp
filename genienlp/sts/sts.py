from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
#
#
# model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
#
# src_sentences = ['I am here']
# trg_sentences1 = ['I am not there']
# trg_sentences2 = ['He is here']
# trg_sentences3 = ['I am there']
#
# for tgt_sent in [trg_sentences1, trg_sentences2, trg_sentences3]:
#
#     dev_mse = evaluation.MSEEvaluator(src_sentences, tgt_sent, name='dev',
#                                       teacher_model=model, batch_size=1)
#
#     score1 = -1.0 * dev_mse(model, output_path='./')
#
#     print('dev_mse', score1)
#
#
#
# src_sentences = ['I am here']
# trg_sentences1 = ['No estoy ahí']
# trg_sentences2 = ['Él está aquí']
# trg_sentences3 = ['Estoy ahí']
#
# for tgt_sent in [trg_sentences1, trg_sentences2, trg_sentences3]:
#
#     dev_trans_acc = evaluation.TranslationEvaluator(src_sentences, tgt_sent, name='dev',
#                                                     batch_size=1, print_wrong_matches=True)
#
#     score2 = dev_trans_acc(model, output_path='./')
#
#     print('dev_trans_acc', score2)


##############################
##############################
# from sentence_transformers.readers import STSBenchmarkDataReader
# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# from sentence_transformers import SentenceTransformer
# import os
#
# model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
#
# script_folder_path = os.path.dirname(os.path.realpath(__file__))
# sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, '../datasets/stsbenchmark'))
#
# evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples("sts-test.csv"), name='sts-test')
#
# model.evaluate(evaluator)

##############################
##############################


from sklearn.metrics.pairwise import paired_cosine_distances

# model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")
# model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
# model = SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")
# model = SentenceTransformer("distilbert-multilingual-nli-stsb-quora-ranking")
model = SentenceTransformer("LaBSE")

src_sentences = ['I am here', 'I am here', 'I am here', 'I am here']
trg_sentences = ['I am not there', 'He is here', 'I am there', 'I am here']
labse_trg_sentences = ['Non sono li', 'Lui è qui', 'Ci sono', 'Sono qui']
labels = [1, 0.5, 0, 1]

batch_size = 4

embeddings1 = model.encode(src_sentences, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True)
embeddings2 = model.encode(labse_trg_sentences, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True)

cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

print('cosine_scores', cosine_scores)

# dev_mse = evaluation.EmbeddingSimilarityEvaluator(src_sentences, trg_sentences1, scores=)

# score1 = -1.0 * dev_mse(model, output_path='./')

# print('dev_mse', score1)
