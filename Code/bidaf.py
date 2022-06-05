from keras.layers import Input, TimeDistributed, LSTM, Bidirectional
from keras.models import Model, load_model
from keras.optimizers import Adadelta
from keras.callbacks import CSVLogger, ModelCheckpoint
from ..layers import Highway, Similarity, C2QAttention, Q2CAttention, MergedContext, SpanBegin, SpanEnd, CombineOutputs
from ..scripts import negative_avg_log_error, accuracy, tokenize, MagnitudeVectors, get_best_span, \
    get_word_char_loc_mapping
from ..scripts import ModelMGPU
import os
import tokenization
from bert_serving.client import BertClient
import numpy as np

def letstry(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class BidirectionalAttentionFlow():

    def __init__(self, emdim, max_passage_length=None, max_query_length=None, num_highway_layers=2, num_decoders=1,
                 encoder_dropout=0, decoder_dropout=0):
        self.emdim = emdim
        # print(self.emdim)
        self.max_passage_length = max_passage_length
        self.max_query_length = max_query_length

        # passage_input = Input(shape=(self.max_passage_length, emdim), dtype='float32', name="passage_input")
        # question_input = Input(shape=(self.max_query_length, emdim), dtype='float32', name="question_input")

        # question_embedding = question_input
        # passage_embedding = passage_input
        # for i in range(num_highway_layers):
        #     highway_layer = Highway(name='highway_{}'.format(i))
        #     question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
        #     question_embedding = question_layer(question_embedding)
        #     passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
        #     passage_embedding = passage_layer(passage_embedding)

        # encoder_layer = Bidirectional(LSTM(emdim, recurrent_dropout=encoder_dropout,
        #                                    return_sequences=True), name='bidirectional_encoder')
        # encoded_question = encoder_layer(question_embedding)
        # encoded_passage = encoder_layer(passage_embedding)

        encoded_question = Input(shape=(self.max_query_length, emdim), dtype='float32', name="question_input")
        encoded_passage = Input(shape=(self.max_passage_length, emdim), dtype='float32', name="passage_input")

        similarity_matrix = Similarity(name='similarity_layer')([encoded_passage, encoded_question])

        context_to_query_attention = C2QAttention(name='context_to_query_attention')([
            similarity_matrix, encoded_question])
        query_to_context_attention = Q2CAttention(name='query_to_context_attention')([
            similarity_matrix, encoded_passage])

        merged_context = MergedContext(name='merged_context')(
            [encoded_passage, context_to_query_attention, query_to_context_attention])

        modeled_passage = merged_context
        for i in range(num_decoders):
            hidden_layer = Bidirectional(LSTM(emdim, recurrent_dropout=decoder_dropout,
                                              return_sequences=True), name='bidirectional_decoder_{}'.format(i))
            modeled_passage = hidden_layer(modeled_passage)

        span_begin_probabilities = SpanBegin(name='span_begin')([merged_context, modeled_passage])
        span_end_probabilities = SpanEnd(name='span_end')(
            [encoded_passage, merged_context, modeled_passage, span_begin_probabilities])

        output = CombineOutputs(name='combine_outputs')([span_begin_probabilities, span_end_probabilities])

        model = Model([encoded_passage, encoded_question], [output])

        model.summary()

        try:
            model = ModelMGPU(model)
        except:
            print("using single gpu")
            pass
        # # model = ModelMGPU(model, 4)

        model.summary()

        adadelta = Adadelta(lr=0.05)
        model.compile(loss=negative_avg_log_error, optimizer=adadelta, metrics=[accuracy])

        self.model = model

    def load_bidaf(self, path):
        custom_objects = {
            # 'Highway': Highway,
            'Similarity': Similarity,
            'C2QAttention': C2QAttention,
            'Q2CAttention': Q2CAttention,
            'MergedContext': MergedContext,
            'SpanBegin': SpanBegin,
            'SpanEnd': SpanEnd,
            'CombineOutputs': CombineOutputs,
            'negative_avg_log_error': negative_avg_log_error,
            'accuracy': accuracy
        }

        self.model = load_model(path, custom_objects=custom_objects)

    def train_model(self, train_generator, steps_per_epoch=None, epochs=1, validation_generator=None,
                    validation_steps=None, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0,
                    save_history=False, save_model_per_epoch=False):
        print("value of use_multiprocessing" + str(use_multiprocessing))
        saved_items_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'saved_items')
        if not os.path.exists(saved_items_dir):
            os.makedirs(saved_items_dir)

        callbacks = []

        if save_history:
            history_file = os.path.join(saved_items_dir, 'history')
            csv_logger = CSVLogger(history_file, append=True)
            callbacks.append(csv_logger)

        if save_model_per_epoch:
            save_model_file = os.path.join(saved_items_dir, 'bidaf_{epoch:02d}.h5')
            checkpointer = ModelCheckpoint(filepath=save_model_file, verbose=1)
            callbacks.append(checkpointer)

        history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                           callbacks=callbacks, validation_data=validation_generator,
                                           validation_steps=validation_steps, workers=workers,
                                           use_multiprocessing=use_multiprocessing, shuffle=shuffle,
                                           initial_epoch=initial_epoch)
        if not save_model_per_epoch:
            self.model.save(os.path.join(saved_items_dir, 'bidaf.h5'))

        return history, self.model

    # def tokenize(sequence, do_lowercase):
    #     """Tokenizes the input sequence using nltk's word_tokenize function, replaces two single quotes with a double quote"""

    #     if do_lowercase:
    #         tokens = [token.replace("``", '"').replace("''", '"').lower()
    #                   for token in nltk.word_tokenize(sequence)]
    #     else:
    #         tokens = [token.replace("``", '"').replace("''", '"')
    #                   for token in nltk.word_tokenize(sequence)]
    #     return tokens


    def predict_ans(self, passage, question, squad_version=1.1, max_span_length=384, do_lowercase=True,
                    return_char_loc=False, return_confidence_score=False):

        tokenizer = tokenization.FullTokenizer(vocab_file='./scibert_scivocab_uncased/vocab.txt', do_lower_case=True)

        passage = passage.strip()

        passage = passage.replace("''", '" ')
        passage = passage.replace("``", '" ')

        self.bc = BertClient()
        res = [question + " ||| " + passage]

        get_encoding = self.bc.encode(res, show_tokens=True)

        get_encoding_zero = np.array(get_encoding[0])

        get_encoding_zero = get_encoding_zero[0][1:-1]

        get_encoding_one = get_encoding[1][0][1:-1]

        question_batch_old = []
        context_batch_old = []

        temp = get_encoding_one.index('[SEP]')
        question_batch_old = get_encoding_zero[0:temp, :]
        context_batch_old = get_encoding_zero[temp+1:, :]

        question_batch = np.expand_dims(np.array(question_batch_old), axis=0)
        context_batch = np.expand_dims(np.array(context_batch_old), axis=0)

        y = self.model.predict([context_batch, question_batch])
        y_pred_start = y[:, 0, :]
        y_pred_end = y[:, 1, :]

        # print(y_pred_start)

        # clearing the session releases memory by removing the model from memory
        # using this, you will need to load model every time before prediction
        # K.clear_session()

        original_passage = [passage, ]
        contexts = context_batch
        batch_answer_span = []
        batch_confidence_score = []
        for sample_id in range(len(contexts)):
            answer_span, confidence_score = get_best_span(y_pred_start[sample_id, :], y_pred_end[sample_id, :],
                                                          len(contexts[sample_id]), squad_version, len(contexts[sample_id]))
            batch_answer_span.append(answer_span)
            batch_confidence_score.append(confidence_score)

        answers = []
        for index, answer_span in enumerate(batch_answer_span):
            original_tokens_try = tokenize(passage, do_lowercase)
            orig_to_tok_map = []
            context_tokens = []
            for orig_token in original_tokens_try:
                orig_to_tok_map.append(len(context_tokens))
                context_tokens.extend(tokenizer.tokenize(orig_token))

            # context_tokens = get_encoding_one[temp+1:]
            # print(context_tokens)
            start, end = answer_span[0], answer_span[1]
            # print("hello" + str(start))
            # start = trialf(orig_to_tok_map, start)

            # start = letstry(orig_to_tok_map, start)
            # print(original_passage[index].lower())
            # print(original_tokens_try)
            start, end = orig_to_tok_map.index(letstry(orig_to_tok_map, start)), orig_to_tok_map.index(letstry(orig_to_tok_map, end))
            # print(start)
            # print(end)
            # word index to character index mapping

            # print(context_tokens)
            # print(len(context_tokens))
            mapping = get_word_char_loc_mapping(original_passage[index].lower(), original_tokens_try)
            # print(mapping)
            char_loc_start = mapping[start]
            # [1] => char_loc_end is set to point to one more character after the answer
            char_loc_end = mapping[end] + len(original_tokens_try[end])
            # [1] will help us getting a perfect slice without unnecessary increments/decrements
            ans = original_passage[index][char_loc_start:char_loc_end]

            return_dict = {
                "answer": ans,
            }

            if return_char_loc:
                return_dict["char_loc_start"] = char_loc_start
                return_dict["char_loc_end"] = char_loc_end - 1

            if return_confidence_score:
                return_dict["confidence_score"] = batch_confidence_score[index]

            answers.append(return_dict)

        return answers[0]
